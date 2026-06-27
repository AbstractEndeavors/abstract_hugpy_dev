from .imports import *

class ChatBody(BaseModel):
    model_key: Optional[str] = None
    prompt: Optional[str] = None
    messages: Optional[List[dict]] = None
    file: Optional[str] = None          # server path from /api/uploads
    images: Optional[List[str]] = None  # base64, if you also do inline images
    # None = "as many as the model allows" — resolved to the model's context at
    # request time. The worker also auto-continues past this per-call cap, so a
    # response is never truncated by the token budget.
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    do_sample: Optional[bool] = None
    # False forces a bounded single pass even when max_new_tokens is omitted.
    unbounded: Optional[bool] = None
    # Explicit dispatch task key (e.g. "image-text-to-text"); wins over the
    # model's primary_task and over the text-only auto-routing below.
    task: Optional[str] = None
    # Client-supplied id so a chat can be cancelled mid-stream
    # (POST /api/llm/chat/cancel/<request_id>).
    request_id: Optional[str] = None
    # Dedicated worker pool override. The route resolves the effective pool from
    # the API key's bound pool (default) + this field (override, if the key
    # allows it). "" / None = the general pool.
    pool: Optional[str] = None

    @model_validator(mode="after")
    def _require_one_input(self):
        if not self.prompt and not self.messages:
            raise ValueError("ChatBody needs either 'prompt' or 'messages'")
        return self
    
class Message(BaseModel):
    role: str
    content: str
    images: List[str] | None = None
    file: str | None = None     # server path from /api/uploads
