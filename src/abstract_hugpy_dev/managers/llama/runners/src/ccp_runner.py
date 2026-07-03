from .base_runner import LlamaCppBaseRunner
from .imports import *
from ...serve import serve_endpoint, serve_model_name

def __init__(self, model_key, *, env_path=None):
    base = serve_endpoint(model_key)          # None for mode=off
    if not base:
        raise RuntimeError(f"{model_key}: no HTTP endpoint (mode=off) — use in-process")
    self.model_key = model_key
    self.base_url = base
    self.served_model = serve_model_name(model_key)
# ===========================================================================
# HTTP runner — talks to a running llama-server process
# ===========================================================================


def _model_is_vision(model_key: str) -> bool:
    """True for an image-text-to-text GGUF — so the base runner folds the image
    into the request (_attach_image) before forwarding to the native
    llama-server --mmproj, which DOES understand image_url content. Without this
    the HTTP runner sends text-only and the server is blind to the image."""
    try:
        from .....imports.config.main import get_model_config
        cfg = get_model_config(model_key)
    except Exception:
        return False
    if getattr(cfg, "primary_task", None) == "image-text-to-text":
        return True
    return "image-text-to-text" in (getattr(cfg, "tasks", None) or [])


class LlamaCppRunner(LlamaCppBaseRunner):
    def __init__(self, model_key: str, *, env_path: Optional[str] = None,
                 base_url: Optional[str] = None):
        # Vision GGUFs: fold the image into the request (gated on is_vision in
        # _attach_image). The native llama-server --mmproj sees image_url content.
        self.is_vision = _model_is_vision(model_key)
        # Explicit base_url: a managed server we were handed (e.g. a shard
        # lead from ensure_shard_server) rather than the serve layer's.
        if base_url:
            self.model_key = model_key
            self.base_url = base_url.rstrip("/")
            self.served_model = model_key
            return

        base = serve_endpoint(model_key)          # None for mode=off
        if not base:
            raise RuntimeError(f"{model_key}: no HTTP endpoint (mode=off) — use in-process")
        self.model_key = model_key
        self.base_url = base
        self.served_model = serve_model_name(model_key)

##        cfg = load_llama_config(env_path=env_path)
##        self.model_key = model_key
##        self.llama_host: str = cfg["LLAMA_HOST"]
##        self.port: int = cfg[model_key]
##        self.base_url = f"{self.llama_host}:{self.port}"

    def _refresh_endpoint(self) -> bool:
        """Re-resolve the serving endpoint after a stale-slot failure.

        This runner instance is CACHED by dispatch, so its base_url can
        outlive the slot it points at (the slot self-heals/unloads/TTLs the
        model, the agent restarts, …) — the proxy then 503s "no model
        loaded" forever while the cached instance keeps knocking. serve
        resolution is authoritative and side-effecting (it loads the model
        into a free slot and waits healthy), so one refresh + retry turns a
        permanently-broken cached runner into a slow first request."""
        try:
            base = serve_endpoint(self.model_key)
        except Exception:
            return False
        if not base:
            return False
        # SAME URL is still a successful refresh: serve resolution is
        # side-effecting — it (re)loads the model into the slot and WAITS for
        # it to go healthy — so retrying the identical endpoint after it
        # returns is exactly right (post-reexec cold window, mid-reload).
        self.base_url = base.rstrip("/")
        self.served_model = serve_model_name(self.model_key)
        return True

    async def _iter_stream(self, messages, max_tokens, temp, top_p):
        payload = {"messages": messages, "max_tokens": max_tokens,
                   "temperature": temp, "top_p": top_p, "stream": True}
        for attempt in (1, 2):
            try:
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream("POST", f"{self.base_url}/v1/chat/completions",
                                             json=payload) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if not line or line.strip() == "[DONE]":
                                continue
                            line = line.removeprefix("data: ")
                            try:
                                data = json.loads(line)
                                choice = data["choices"][0]
                                text = (choice.get("delta") or {}).get("content") or ""
                                fr   = choice.get("finish_reason")
                            except Exception:
                                text, fr = "", None
                            yield text, fr
                return
            except (httpx.HTTPStatusError, httpx.ConnectError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if attempt == 1 and (status in (502, 503) or isinstance(exc, httpx.ConnectError)):
                    # Stale cached endpoint (slot unloaded/agent restarted):
                    # re-resolve — serve_endpoint reloads the model — and retry once.
                    if self._refresh_endpoint():
                        continue
                raise
    def _chat_complete(self, messages, max_tokens, temp, top_p, stop):
        payload = {"messages": messages, "max_tokens": max_tokens,
                   "temperature": temp, "top_p": top_p, "stream": False}
        if stop:
            payload["stop"] = stop
        for attempt in (1, 2):
            try:
                with httpx.Client(timeout=DEFAULT_HTTP_TIMEOUT) as client:
                    r = client.post(f"{self.base_url}/v1/chat/completions", json=payload)
                    r.raise_for_status()
                    data = r.json()
                break
            except (httpx.HTTPStatusError, httpx.ConnectError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if attempt == 1 and (status in (502, 503) or isinstance(exc, httpx.ConnectError)):
                    if self._refresh_endpoint():   # stale cached endpoint — see _iter_stream
                        continue
                raise
        choice = data["choices"][0]
        return choice["message"]["content"] or "", choice.get("finish_reason") or "stop"

    def _raw_complete(self, prompt, max_tokens, temp, top_p, stop, return_full_text):
        payload = {"prompt": prompt, "n_predict": max_tokens,
                   "temperature": temp, "top_p": top_p, "stream": False}
        if stop:
            payload["stop"] = stop
        with httpx.Client(timeout=DEFAULT_HTTP_TIMEOUT) as client:
            r = client.post(f"{self.base_url}/completion", json=payload)
            r.raise_for_status()
            data = r.json()
        text = data.get("content") or data.get("text") or ""
        if return_full_text:
            text = prompt + text
        finish = data.get("stop_type") or data.get("finish_reason") or "stop"
        return text, finish
    def _blocking_complete(
        self,
        messages: list[dict] | str,
        max_tokens: int,
        temp: float,
        top_p: float,
        stop: Optional[list[str]],
        use_chat_template: bool,
        return_full_text: bool,
    ) -> tuple[str, str]:
        timeout = DEFAULT_HTTP_TIMEOUT

        if use_chat_template and isinstance(messages, list):
            payload = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temp,
                "top_p": top_p,
                "stream": False,
            }

            if stop:
                payload["stop"] = stop

            with httpx.Client(timeout=timeout) as client:
                response = client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
            finish = choice.get("finish_reason") or "stop"

            return text, finish

        # Raw-prompt fallback: llama-server /completion endpoint
        prompt = (
            messages
            if isinstance(messages, str)
            else messages_to_prompt_from_dicts(messages)
        )

        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temp,
            "top_p": top_p,
            "stream": False,
        }

        if stop:
            payload["stop"] = stop

        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{self.base_url}/completion",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        text = data.get("content") or data.get("text") or ""

        if return_full_text:
            text = prompt + text

        finish = data.get("stop_type") or data.get("finish_reason") or "stop"

        return text, finish
