from .imports import *
from .task_schemas import *
from pydantic import model_validator, model_serializer
ChatInput = Union["ChatRequest", Mapping, str]  # request | dict-ish | bare prompt

class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: ROLES = "user"
    # content is optional on the OpenAI tool-calling shapes: an assistant turn
    # that ONLY calls a tool carries `tool_calls` and a null `content`. Kept
    # str|None (default "") so plain chat is byte-for-byte unchanged.
    content: Optional[str] = ""
    # OpenAI tool-calling fields (the /v1 loop, step 2+). Added explicitly so
    # the schema stays extra="forbid": an assistant echo-back carries its
    # `tool_calls` array, and a `{"role":"tool"}` result carries `tool_call_id`.
    # The runner never sees these — v1_helpers renders them into `content` text
    # (the Qwen <tool_call>/<tool_response> convention) before relay, so a
    # released worker on the frozen role+content wire keeps working.
    tool_calls: Optional[list] = None
    tool_call_id: Optional[str] = None

    @model_serializer(mode="wrap")
    def _omit_null_tool_fields(self, handler):
        # Workers on the released package re-validate the relayed request with
        # a ChatMessage that is still role+content only and extra="forbid" —
        # dumping tool_calls/tool_call_id as None keys breaks EVERY offloaded
        # chat. Omit them unless actually set so the relay wire stays
        # byte-identical to the frozen schema for all non-tool traffic.
        data = handler(self)
        for key in ("tool_calls", "tool_call_id"):
            if data.get(key) is None:
                data.pop(key, None)
        return data

class ChatRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    request_id: str = Field(default_factory=lambda: get_request_id())
    model_key: str = None
    messages: list[ChatMessage]
    max_new_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    do_sample: bool = False
    unbounded: bool = False
    max_chunks: Optional[int] = None
    file: Optional[str] = None
    # Inline images for the current turn as base64 (raw or full data: URI). The
    # no-upload vision path: the runner folds these into the latest user turn as
    # image_url parts (see LlamaCppBaseRunner._attach_image). None for text chat.
    images: Optional[list[str]] = None
    # Dedicated worker pool to route this request to ("" / None = general pool).
    # Resolved at the route from the API key's bound pool + an optional override.
    pool: Optional[str] = None
    @field_validator("messages", mode="before")
    @classmethod
    def normalize_messages(cls, value: Any) -> Any:
        if isinstance(value, str):
            return get_messages(value)
        return value

    @model_validator(mode="before")
    @classmethod
    def _normalize_multimodal(cls, data: Any) -> Any:
        """Funnel every image-bearing message shape down to one path.

        Whatever a client sends — OpenAI-style ``content`` arrays of
        ``{"type":"text"}`` / ``{"type":"image_url"}`` parts, a bare image part,
        ``input_image`` — the image(s) are hoisted into ``images`` (data-URI /
        base64 / url strings) and the message ``content`` is flattened to its
        text. The runner then turns ``images`` into the bytes the model sees, so
        there is a single image path regardless of how the request was phrased.
        """
        if not isinstance(data, Mapping):
            return data
        msgs = data.get("messages")
        if not isinstance(msgs, list):
            return data
        collected: list = list(data.get("images") or [])

        def _img_url(part: Mapping) -> Optional[str]:
            val = part.get("image_url", part.get("url", part.get("image")))
            if isinstance(val, Mapping):
                val = val.get("url")
            return val if isinstance(val, str) and val else None

        out_msgs = []
        for m in msgs:
            if isinstance(m, Mapping) and isinstance(m.get("content"), list):
                m = dict(m)
                texts: list[str] = []
                for part in m["content"]:
                    if not isinstance(part, Mapping):
                        if isinstance(part, str):
                            texts.append(part)
                        continue
                    ptype = part.get("type")
                    if ptype in ("image_url", "input_image", "image"):
                        url = _img_url(part)
                        if url:
                            collected.append(url)
                    elif ptype == "text":
                        texts.append(part.get("text") or "")
                    else:  # unknown part with a usable text/url, be lenient
                        if isinstance(part.get("text"), str):
                            texts.append(part["text"])
                m["content"] = "\n".join(t for t in texts if t)
            out_msgs.append(m)

        data = dict(data)
        data["messages"] = out_msgs
        if collected:
            data["images"] = collected
        return data

    @classmethod
    def coerce(cls, value: ChatInput, *, model_key: Optional[str] = None) -> "ChatRequest":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(model_key=model_key, messages=value)  # validator handles it
        if isinstance(value, Mapping):
            data = dict(value)
            
            if "messages" not in data and "prompt" in data:
                prompt = data.pop("prompt")
                file = data.pop("file", None)
                if file:
                    content = read_from_file(file)
                    prompt = f"{prompt}\n------{file}------\n{content}"
                system = data.pop("system", None)
                msgs = []
                if system:
                    msg = get_message(content=prompt,role="system")
                    msgs.append(msg)
                msg = get_message(content=prompt,role="user")
                msgs.append(msg)
                data["messages"] = msgs
            if "model_key" not in data and model_key:
                data["model_key"] = model_key
            return cls.model_validate(data)
        raise TypeError(f"cannot coerce {type(value).__name__} to ChatRequest")

class ChatResult(TaskResult):
    text: str
    finish_reason: FINISH_REASONS
    usage: Optional[dict] = None
    output_chunks: int = 0
