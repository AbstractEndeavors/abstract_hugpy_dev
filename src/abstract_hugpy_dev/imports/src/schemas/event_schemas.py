from .imports import *
class TokenEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["token"] = "token"
    request_id: str
    text: str

class DoneEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["done"] = "done"
    request_id: str
    input_tokens: int
    output_chunks: int
    finish_reason: FINISH_REASONS

class ErrorEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["error"] = "error"
    request_id: str
    message: str

class StatusEvent(BaseModel):
    """Out-of-band passthrough event — provisioning progress, continuation
    segment markers, and anything a remote worker emits that isn't a
    token/done/error. ``extra="allow"`` so it can carry stage/message/progress/
    done_bytes/etc. without a rigid schema; ``type`` defaults to "status" but is
    overwritten when reconstructed from a worker line (e.g. type="request").
    The route serializer just model_dump()s these straight to the SSE wire."""
    model_config = ConfigDict(extra="allow")
    type: str = "status"
    request_id: str = ""
