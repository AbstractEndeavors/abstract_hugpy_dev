# Allocation modes — where your model's memory lives

Every model gets **one of five modes** that says how its weights (and KV
context) are placed across the GPU (VRAM) and system RAM. Pick the intent; the
fleet does the math.

> **The promise:** a *slow* mode you chose is yours; a model with *no working
> mode at all* is our bug. The default never OOMs — a blank model serves.

## The five modes

| mode | what it does | when it refuses ("bust") |
|---|---|---|
| **gpu-only** | Every layer on the GPU, no spill. Fastest when it fits. | Won't fit the GPU (even after evicting idle residents) → refuse. Never a half-load. |
| **ram-only** | Everything in system RAM; the GPU is never touched (binds the CPU even when a GPU is present). | Won't fit RAM → refuse. |
| **max-gpu** | *The default.* As much GPU as is available and needed, the rest spills to RAM. Zero knobs — "use my GPU, spill the rest". | Only when GPU + RAM together can't hold it. |
| **max-ram** | Mirror of max-gpu: fill RAM first, only the overflow rides the GPU. For keeping cards free while big models stay warm. | Only when RAM + GPU together can't hold it. |
| **explicit** | You set the targets: a VRAM and/or RAM budget, a **leniency %**, and a **priority device** (`gpu` default, or `ram`), plus the tolerance bands/priority you already know. | Can't hit the target *even at the loosened floor* → refuse. |

`max-gpu` and `max-ram` are internally explicit-with-generous-leniency — but
they stay flat, named modes on purpose: simple intent gets a simple pick; the
moment you want to tune numbers, you reach for `explicit`.

## Leniency — how much off-device you tolerate

**N% leniency = up to N% of the MODEL may land off its ideal device before we
refuse.**

Example: `100% GPU target + 30% leniency`
- Try 100% on GPU first.
- Under pressure, degrade step by step — 90/10, 80/20 … — down to the **floor:
  70% GPU / 30% RAM**.
- If even 70/30 won't fit, the load refuses **honestly**, naming the mode and
  the floor (e.g. *"explicit: even the loosened floor won't fit — target 100%
  of the model on GPU with 30% leniency gives a floor of 70% …"*).

No leniency set = exact target or refuse (no undeclared tolerance is ever
assumed). Degradation only happens under real contention — a model that fits
at its target always gets its target.

## Engine coverage

| mode | GGUF (llama.cpp) | transformers / diffusers |
|---|---|---|
| gpu-only | ✅ | ✅ (whole model on the card) |
| ram-only | ✅ | ✅ (CPU placement) |
| max-gpu | ✅ | ✅ (accelerate fit-and-spill) |
| max-ram | ✅ | ❌ for now — refused with a clear 409 |
| explicit | ✅ | ❌ for now — refused with a clear 409 |

Fine-grained placement (max-ram/explicit) needs loader wiring the transformers
gap loaders don't have yet; until that ships, picking one for a transformers
model refuses honestly instead of silently doing something else.

## Legacy names (the honest rename)

Old controls were misleadingly named; existing settings keep working — old
names are accepted on input, resolved, and never written back:

| old name / setting | is now | why |
|---|---|---|
| **autofit** ({} / no knobs) | **max-gpu** | it always was "as much GPU as fits, spill the rest" |
| **Max GPU** (`n_gpu_layers: -1`) | **gpu-only** | it was all-or-OOM, not "maximize" — now it refuses instead of crashing |
| **CPU only** (`n_gpu_layers: "off"`) | **ram-only** | RAM is what it actually binds |
| **budget** / **bands** (chaos) | **explicit** | a budget/band IS an explicit allocation |

Nothing on the wire changed for these three: `gpu-only` still rides
`n_gpu_layers: -1`, `ram-only` still rides `"off"`, `max-gpu` is still `{}`.
Your `serve_overrides.json` is never rewritten — the mode is **derived at read
time** (`-1` → gpu-only, `0/"off"` → ram-only, budgets/bands → explicit, unset
→ max-gpu).

## Old workers

`max-ram` and `explicit` ride new spill keys (`alloc_mode`, `leniency_pct`,
`priority_device`). Central only emits them to workers running **0.1.203+**;
an older worker gets max-gpu (autofit) for that request and the downgrade is
logged — your persisted choice is kept and applies the moment the worker
updates. A selected mode is never a silent dead knob.

## Protection is unchanged

Modes reorder *placement*, never protection: 🔒 static residents and models
actively answering are never evicted to make room, whatever mode the incoming
model carries.
