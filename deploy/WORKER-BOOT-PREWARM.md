# Per-worker KEEP-WARM STAR (`boot_prewarm`)

Operator RULINGS 2026-07-23:

- **RULING 1** — "the star is the ONLY warm source. **nothing warms until
  starred (or static).**"
- **RULING 2** — "**star = reconcile-kept-warm**" (NOT boot-once): a starred
  model that gets evicted under load **comes back on the next reconcile beat**.
  The ⭐ star **is** the keep-warm designation.

A worker carries a single ⭐ **star** model it **keeps warm**. The identifier
stays `boot_prewarm`/`boot-prewarm` (rename churn isn't worth it) but the
meaning is **keep-warm**, not boot-once.

## The three levers (locked semantics)

| lever | warms? | eviction | notes |
|---|---|---|---|
| ⭐ **star** (`boot_prewarm`) | **Yes** — reconcile keeps it warm **every beat** | **Evictable under pressure, but RETURNS next reconcile cycle** | NOT eviction-protected. The star IS the keep-warm designation. |
| 🔒 **static** | **Yes** | **Protected** — never evicted by the LLM plane | "start here AND stay here" — unchanged. |
| 📌 **pin** | **No** | n/a | Routing persistence only — never warms. Unchanged. |

And for contrast, the /media star:

| | media_default (the /media star) | ⭐ boot_prewarm (this per-worker star) |
|---|---|---|
| Scope | one global model | one model **per worker** |
| Effect | "first in the list + default-selected" — a **UI/routing preference** | "**keep this model warm** on this worker" |
| Loads anything? | **No** | **Yes — reconcile-kept warm** |

**Nothing warms until starred or static.** The fleet `TASK_DEFAULTS` (sd-turbo
et al.) are a **routing fallback** — a request naming only a task still resolves
to its default model (`model_resolver.TASK_DEFAULTS`) — but they are **no longer
kept warm**. 📌 pins never warm. Everything outside (⭐ star ∪ 🔒static) lazy-loads
on first real request.

### How keep-warm works

- **Central side** — `_reconcile_warm_set(worker)` = `(⭐ star ∪ 🔒static) ∩
  models_local − blocked`. Every heartbeat, central re-computes this set and
  re-warms any cold member (rate-limited by `HUGPY_WARM_COOLDOWN_S`, fit-capped
  by `_warmable_subset`). Because it re-runs every beat, an evicted star is
  reloaded next cycle.
- **Worker side** — `_adopt_boot_prewarm` runs on every register/heartbeat
  reply: if the star is **not currently resident** (in-process or slot-seated),
  it kicks a load through the normal on-demand path (no residency write — the
  model stays FIFO-evictable); if it's already loaded, it's a **no-op**. An
  in-flight guard stops two load threads racing for the same star, but it is
  **not** a permanent latch — it clears when the load finishes, so a later beat
  after an eviction re-warms.

**Want eviction protection ("start here AND stay here")?** Promote the model to
🔒static — that tier, not the star, protects residency.

## API

All paths are relative to central (e.g. `https://dev.hugpy.ai/api`). Worker ids
come from `GET /llm/workers` (the `id` field).

- **Set / replace a worker's star** (operator-gated):
  `POST /llm/workers/<worker_id>/boot-prewarm`
  Body: `{"model_key": "<key>", "enabled": true}` — makes `model_key` the star,
  replacing any previous one. `{"enabled": false}` (optionally with a matching
  `model_key`, or none) clears it.

- **Read the full star map** (open, read tier): `GET /llm/workers/boot-prewarm`
  → `{"<worker_id>": "<model_key>", ...}`

- **Per-worker surfacing**: every row of `GET /llm/workers` (and
  `GET /llm/workers/<id>`) carries `"boot_prewarm": "<model_key>"|null`, so the
  console can render the star.

The star rides the register/heartbeat reply to the worker as
`boot_prewarm: "<model_key>"` — additive and **omit-when-unset**, so a released
worker that predates the feature simply ignores it (the `extra=forbid` relay
schema is never broken).

## Seed the operator's two defaults

These are **data, not code** — run them once against central with the operator
token. Replace `<computron-id>` / `<ae-id>` with the ids from `GET /llm/workers`.

```bash
# computron (RTX-4060) -> a present 7B. Use whichever 7B key the fleet actually
# holds (check `GET /llm/workers/<computron-id>` models / `GET /models`):
#   Qwen~Qwen2-7B-Instruct-GGUF   (or Qwen~Qwen2.5-7B-Instruct-GGUF)
curl -fsS -X POST https://dev.hugpy.ai/api/llm/workers/<computron-id>/boot-prewarm \
  -H "X-Operator-Token: $HUGPY_OPERATOR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model_key": "Qwen~Qwen2-7B-Instruct-GGUF", "enabled": true}'

# ae (RTX-3090) -> the agent brain, coder-next (DEFAULT_AGENT_BRAIN):
curl -fsS -X POST https://dev.hugpy.ai/api/llm/workers/<ae-id>/boot-prewarm \
  -H "X-Operator-Token: $HUGPY_OPERATOR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model_key": "Qwen~Qwen3-Coder-Next-GGUF", "enabled": true}'
```

Verify: `GET /llm/workers/boot-prewarm` returns both, and each worker keeps its
star warm every reconcile beat (`keep-warm star … warming …` in the agent log
when it is cold; a no-op when already resident).

### Note on the 7B key for computron

The brief named `Qwen2-7B-Instruct-GGUF` / `Qwen2.5 7B`. Neither is a curated
staple in `models_config.MODELS` (they arrive via discovery), so the exact
`model_key` is whatever the fleet discovered — confirm against `GET /models`
before seeding. The star store does **not** require the model to be present or
allocated when you set it; if the model can't be loaded, the worker logs and
continues (never crashes).
