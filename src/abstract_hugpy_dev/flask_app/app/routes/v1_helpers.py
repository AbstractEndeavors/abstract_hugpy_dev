"""Pure helpers for the OpenAI-compatible /v1 route (v1_routes.py).

Deliberately stdlib-only (json/re/uuid) with NO Flask or package imports, so
they unit-test standalone (tests/test_v1_seam.py loads this file directly by
path). Everything here is request/response plumbing that must be testable
without a running app: payload -> engine kwargs translation, and (later
commits) the tools prompt-shim + <tool_call> parser and usage shaping.

Naming keeps the leading underscore the route always used — these are private
to the /v1 seam, the module split exists only for offline testability.
"""
from __future__ import annotations

import json
import re
import uuid


def _completion_kwargs(payload: dict) -> dict:
    """Translate an OpenAI chat.completions payload into engine prompt_kwargs.

    Only fields the frozen ChatRequest schema (extra="forbid") defines are
    forwarded — anything else must stay route-local (e.g. `tools`, handled by
    the prompt shim in v1_routes, never reaches the engine).
    """
    messages = payload.get("messages")
    if not messages:
        raise ValueError("'messages' is required")
    # OpenAI clients must send *something* as model; "default" (and empty)
    # mean "no preference" — leave model_key unset so resolve() falls through
    # to the reconciled chat default instead of 404ing on a literal "default".
    model = payload.get("model")
    if isinstance(model, str) and model.strip().lower() in ("", "default"):
        model = None
    kwargs = {
        "messages": [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in messages
        ],
        "model_key": model,
        "request_id": f"v1-{uuid.uuid4().hex}",
    }
    max_tokens = payload.get("max_tokens") or payload.get("max_completion_tokens")
    if max_tokens:
        # Explicit client cap → bounded; omitted → engine runs unbounded with
        # auto-continuation, same as the console.
        kwargs["max_new_tokens"] = int(max_tokens)
    if payload.get("temperature") is not None:
        kwargs["temperature"] = float(payload["temperature"])
        kwargs["do_sample"] = float(payload["temperature"]) > 0
    if payload.get("top_p") is not None:
        kwargs["top_p"] = float(payload["top_p"])
    if payload.get("unbounded") is not None:
        kwargs["unbounded"] = bool(payload["unbounded"])
    # Continuation budget. Never forwarding this was the production stall
    # (2026-07-14): every /v1 request ran with unbounded auto-continuation, so
    # a rambling small model kept getting "Continue exactly where you left off"
    # passes appended until the read timeout. OpenAI `max_tokens` semantics are
    # ONE bounded completion — so a request that caps tokens but doesn't ask
    # for continuation defaults to a single pass. A request with neither cap
    # keeps today's unbounded behavior (the console relies on it).
    if payload.get("max_chunks") is not None:
        kwargs["max_chunks"] = int(payload["max_chunks"])
    elif max_tokens:
        kwargs["max_chunks"] = 1
    return kwargs


def _usage_block(usage) -> dict:
    """Shape a DoneEvent usage dict into the OpenAI `usage` object.

    Real counts when the engine/runner reported them; the historical all-None
    shape when genuinely unavailable (old worker builds, error paths) — a
    missing count must never crash or omit the key, OpenAI SDKs expect it.
    """
    if not isinstance(usage, dict):
        usage = {}

    def _int(key):
        v = usage.get(key)
        return v if isinstance(v, int) else None

    prompt = _int("prompt_tokens")
    completion = _int("completion_tokens")
    total = _int("total_tokens")
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": total}


# ──────────────────────────────────────────────────────────────────────────
# tools shim — prompt-inject + parse, entirely route-local.
#
# The engine schema is frozen (extra="forbid") and GGUF models have no native
# function-calling wire, so /v1 `tools` support lives here: render the tool
# JSON-schemas into a system-prompt preamble using the Qwen2.5/Hermes
# convention (the fleet's Qwen-family GGUFs were trained on exactly this
# format), then parse `<tool_call>{...}</tool_call>` blocks out of the reply.
# Modeled on the sibling client hugpy_agent/adapter.py (prompted tier) so both
# sides of the seam speak the same dialect. `tools` itself NEVER reaches the
# engine.
# ──────────────────────────────────────────────────────────────────────────

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

# Known engine leak strings (the auto-continuation prompts from
# dispatch.execute_chat_stream / hugpy_agent's captured variant). Scrubbed
# before parsing — a leak INSIDE a JSON block would corrupt it — and defense
# in depth on top of the max_chunks=1 forced for tool requests.
_CONTINUATION_LEAKS = (
    "Continue exactly where you left off. Do not repeat any previous text.",
    "Continue exactly where I left off. Do not repeat any previous text.",
    "Continue exactly where you left off.",
)

_TOOLS_PREAMBLE_TEMPLATE = """\
# Tools

You may call ONE function per reply to assist with the task.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_lines}
</tools>

For a function call, return a json object with function name and arguments \
within <tool_call></tool_call> XML tags, then STOP:
<tool_call>
{{"name": "<function-name>", "arguments": {{<args-json-object>}}}}
</tool_call>

The function result will come back inside <tool_response></tool_response> tags.
Never invent a function result — wait for the real one."""


def _build_tools_preamble(tools, tool_choice=None):
    """System-prompt block advertising the request's tools, or None.

    None means "run a plain completion": empty/malformed tools, tool_choice
    "none", or a forced {"function":{"name":...}} that matches nothing. A
    specific forced choice narrows the advertised list to that one tool and
    appends a MUST-call instruction; "auto"/absent advertises them all.
    """
    if not tools or not isinstance(tools, (list, tuple)) or tool_choice == "none":
        return None
    forced = None
    if isinstance(tool_choice, dict):
        forced = ((tool_choice.get("function") or {}).get("name")) or None
    lines = []
    for t in tools:
        fn = (t or {}).get("function") or {} if isinstance(t, dict) else {}
        name = fn.get("name")
        if not name or (forced and name != forced):
            continue
        lines.append(json.dumps({
            "type": "function",
            "function": {
                "name": name,
                "description": fn.get("description") or "",
                "parameters": fn.get("parameters")
                              or {"type": "object", "properties": {}},
            },
        }, separators=(",", ":"), ensure_ascii=False))
    if not lines:
        return None
    preamble = _TOOLS_PREAMBLE_TEMPLATE.format(tool_lines="\n".join(lines))
    if forced:
        preamble += (f"\n\nYou MUST call the function \"{forced}\" now — "
                     "reply with exactly one <tool_call> block.")
    return preamble


def _inject_tools_preamble(messages, preamble):
    """Messages with the tools preamble merged in as/into a system message.

    Appended to an existing leading system message (operator instructions keep
    priority) when its content is plain text; otherwise a fresh system message
    is prepended. Input list/dicts are not mutated.
    """
    out = [dict(m) if isinstance(m, dict) else m for m in (messages or [])]
    for m in out:
        if isinstance(m, dict) and m.get("role") == "system":
            content = m.get("content")
            if content is None or isinstance(content, str):
                m["content"] = f"{content}\n\n{preamble}".strip() if content else preamble
                return out
            break  # multimodal/odd system content — don't corrupt it
    return [{"role": "system", "content": preamble}] + out


def _parse_call_json(block: str, *, require_arguments: bool = False):
    """One tool-call JSON object -> {"name": str, "arguments": dict} or None.

    Tolerates the sloppy-small-model cases the hugpy_agent adapter does:
    double-encoded `arguments` strings are unwrapped one level, and a missing
    `arguments` defaults to {} (unless require_arguments — the bare-JSON
    scan uses that to avoid claiming arbitrary {"name": ...} prose objects).
    """
    try:
        data = json.loads(block)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict) or not data.get("name"):
        return None
    if require_arguments and "arguments" not in data:
        return None
    args = data.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, ValueError):
            return None
    if not isinstance(args, dict):
        return None
    return {"name": str(data["name"]), "arguments": args}


def _bare_call(text: str):
    """First standalone {"name":..., "arguments":...} object in free text.

    Balanced-brace scan (nested braces defeat any regex). Stricter than the
    fenced path: `arguments` must be present, so a prose JSON object that
    merely has a "name" key never becomes a false tool call.
    """
    idx = 0
    while True:
        start = text.find("{", idx)
        if start < 0:
            return None
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    if '"name"' in candidate and '"arguments"' in candidate:
                        call = _parse_call_json(candidate, require_arguments=True)
                        if call:
                            return call
                    break
        else:
            return None
        idx = start + 1


def _parse_tool_calls(text: str):
    """Assistant text -> (clean_text, OpenAI tool_calls list | None).

    Finds every <tool_call>...</tool_call> block (bare-JSON fallback when the
    model skipped the tags). No valid call found — including malformed JSON —
    returns (original text, None): errors-as-data, the caller answers with
    plain content exactly as today rather than 500ing on a shim parse error.
    clean_text is the reply with call blocks and known leak strings removed.
    """
    original = text or ""
    scrubbed = original
    for leak in _CONTINUATION_LEAKS:
        scrubbed = scrubbed.replace(leak, "")

    calls = []
    remaining = scrubbed
    for m in _TOOL_CALL_RE.finditer(scrubbed):
        call = _parse_call_json(m.group(1))
        if call:
            calls.append(call)
            remaining = remaining.replace(m.group(0), "", 1)
    if not calls:
        call = _bare_call(scrubbed)
        if call:
            calls, remaining = [call], ""
    if not calls:
        return original, None

    tool_calls = [{
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {"name": c["name"],
                     "arguments": json.dumps(c["arguments"], ensure_ascii=False)},
    } for c in calls]
    return remaining.strip(), tool_calls
