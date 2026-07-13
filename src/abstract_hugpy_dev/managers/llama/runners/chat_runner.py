"""Llama.cpp chat runner.

Adapter over the existing LlamaCppPythonRunner / LlamaCppRunner (HTTP).
Both are kept in their per-process singleton cache (get_llama_runner),
so the heavy GGUF load happens once per model_key regardless of how
many adapter wrappers exist.

Constructor signature is uniform with the other runners in _RUNNERS:
    __init__(self, cfg: ModelConfig, **runtime_kwargs)

This matches what dispatch._get_or_build_runner expects, so the
dispatch table can stay a flat (framework, task) -> class mapping
without any per-runner adapter logic.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Optional

from .src import *
from .imports import *
from .get import get_llama_runner

logger = logging.getLogger(__name__)


class LlamaCppChatRunner:
    """Runner for GGUF models loaded in-process via llama_cpp or via HTTP.

    Uses the get_llama_runner() singleton cache so multiple adapter
    wrappers for the same model_key share a single underlying runner
    (which itself holds the loaded GGUF + KV cache + generate_lock).
    """

    request_type = ChatRequest
    result_type = ChatResult

    def __init__(self, cfg, **runtime_kwargs):
        self.cfg = cfg
        # model_key is whatever the registry uses as its key; ModelConfig
        # exposes it as model_key (set by get_models_dict in models_config).
        self.model_key = cfg.model_key
        # **runtime_kwargs is accepted to keep the uniform _RUNNERS constructor
        # signature, but deliberately NOT stored or applied: the underlying GGUF
        # runner is a per-model_key singleton (get_llama_runner), so per-call
        # n_ctx/n_threads overrides can't be honored without forcing a second
        # load. GPU/context placement is resolved once, from env, in spill.py.
        if runtime_kwargs:
            logger.debug("LlamaCppChatRunner ignoring runtime_kwargs %s for %s "
                         "(singleton runner; placement comes from spill.py)",
                         sorted(runtime_kwargs), self.model_key)

    @property
    def runner(self):
        # Lazy resolution. First access triggers the GGUF load (which can
        # take seconds for a 14B model); subsequent accesses are dict lookups.
        return get_llama_runner(self.model_key)

    def ensure_loaded(self):
        """Force the underlying GGUF runner to MATERIALIZE now.

        __init__ and runner_for() build only this lazy wrapper — the heavy
        runner (which seats a llama-server slot via get_llama_runner ->
        _build_runner -> SlotPool.endpoint_for, or loads in-process) is
        resolved on first .runner access. Warm / slot-fill / probe paths call
        this so the model actually becomes resident + slot-seated instead of a
        hollow shell that still registers as "loaded". Idempotent — the heavy
        runner is a per-model_key singleton (get_llama_runner cache)."""
        return self.runner

    # --- non-streaming -----------------------------------------------------

    async def run(self, req) -> ChatResult:
        req = ChatRequest.coerce(req, model_key=self.model_key)
        runner = self.runner
        messages = [
            m.model_dump() if hasattr(m, "model_dump") else m
            for m in req.messages
        ]

        # Vision GGUFs: fold the image into the latest user turn as an image_url
        # part. _attach_image was only wired into the STREAMING path, so the
        # non-streaming run() that /ml/vision uses silently dropped the image —
        # the model saw "describe this image" with no image and asked for one.
        has_image = bool(getattr(req, "file", None) or getattr(req, "images", None)) \
            and getattr(runner, "is_vision", False)
        if has_image:
            messages = runner._attach_image(messages, req)

        if req.unbounded and not has_image:
            text = await runner.generate_text_async(
                messages,
                temperature=req.temperature,
                top_p=req.top_p,
                do_sample=req.do_sample,
            )
        else:
            # Image turns MUST go through the chat-template/chat-completion path
            # so the multimodal handler sees the image_url parts; the raw-prompt
            # path (used by unbounded text) flattens messages and drops them.
            text = await runner.generate_text_async(
                messages,
                max_new_tokens=req.max_new_tokens or 512,
                temperature=req.temperature,
                top_p=req.top_p,
                do_sample=req.do_sample,
                use_chat_template=True,
                return_full_text=False,
            )

        return ChatResult(
            request_id=req.request_id,
            model_key=req.model_key,
            ok=True,
            text=text,
            finish_reason="stop",
        )

    # --- streaming ---------------------------------------------------------

    async def stream(
        self,
        req: ChatRequest,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Pick stream_chat or stream_chat_unbounded based on req.unbounded.

        Both methods already conform to the StreamEvent contract
        (TokenEvent stream + one terminal DoneEvent/ErrorEvent), so the
        adapter is a straight passthrough.
        """
        # Per-request loop bounds (bug B): the unbounded continue-loop used to
        # drop the caller's caps entirely. Mirror the DeepCoder path in
        # generate_runner._run/stream unbounded, which threads max_chunks and uses
        # `req.max_new_tokens or 1024` as the per-pass chunk budget. ChatRequest
        # here differs from that mental model in one way that matters: its
        # max_new_tokens is NEVER None — it defaults to DEFAULT_MAX_TOKENS — so a
        # literal `req.max_new_tokens or 1024` would silently change the historical
        # default per-pass budget from 1024 to 32768. To honor the "preserve the
        # unbounded default; only make the caps honorable when a caller SETS them"
        # rule, treat the schema default as "unset" (keep the runner's own 1024)
        # and honor any explicit, non-default value. req.max_chunks is genuinely
        # Optional[None] -> pass through; None lets base_runner apply its
        # HUGPY_MAX_CHUNKS ceiling (256), so the default is unchanged.
        if req.max_new_tokens and req.max_new_tokens != DEFAULT_MAX_TOKENS:
            chunk_tokens = req.max_new_tokens
        else:
            chunk_tokens = 1024
        streamer = (
            self.runner.stream_chat_unbounded(
                req,
                cancel_event=cancel_event,
                chunk_tokens=chunk_tokens,
                max_chunks=req.max_chunks,
            )
            if req.unbounded
            else self.runner.stream_chat(req, cancel_event=cancel_event)
        )
        async for event in streamer:
            yield event
