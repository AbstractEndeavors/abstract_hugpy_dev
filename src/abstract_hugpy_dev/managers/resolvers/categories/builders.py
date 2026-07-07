from .imports import *

# ---------------------------------------------------------------------------
# Request builders — one per (framework, task).
# ---------------------------------------------------------------------------

def _file_as_chat_text(file: str) -> str:
    """A file attachment rendered as chat text, or a clear refusal.

    text-generation models only consume text, so anything else (image, audio,
    video) is rejected here with the routing hint — the same answer whether the
    caller sent 'prompt' or 'messages'.
    """
    media = derive_media_type(file)
    if media not in ("text", "document", "code"):
        raise ValueError(
            f"text-generation can't consume a {media!r} file "
            f"({os.path.basename(file)}); route it to the matching model"
        )
    content = read_from_file(file)
    return f"------ {os.path.basename(file)} ------\n{content}"


def _build_chat_request(kwargs: Dict[str, Any], model_key: str) -> ChatRequest:
    out: Dict[str, Any] = {"model_key": model_key}
    file = kwargs.get("file")

    if "messages" in kwargs:
        messages = [dict(m) if isinstance(m, dict) else m for m in kwargs["messages"]]
        if file:
            # The attachment belongs to the latest user turn (the chat UI sends
            # the whole history plus the file each round).
            blob = _file_as_chat_text(file)
            for m in reversed(messages):
                if isinstance(m, dict) and m.get("role", "user") == "user":
                    m["content"] = f"{m.get('content') or ''}\n{blob}"
                    break
            else:
                messages.append({"role": "user", "content": blob})
        out["messages"] = messages
    elif "prompt" in kwargs:
        prompt = kwargs["prompt"] or ""
        if file:
            prompt = f"{prompt}\n{_file_as_chat_text(file)}"
        out["messages"] = [{"role": "user", "content": prompt}]
    else:
        raise ValueError(
            "chat request needs either 'messages' or 'prompt'; "
            f"got keys: {sorted(kwargs)}"
        )

    for k in ("max_new_tokens", "temperature", "top_p", "do_sample", "request_id",
              "unbounded", "pool", "images"):
        if k in kwargs:
            out[k] = kwargs[k]
    out.setdefault("request_id", make_request_id())
    # Default chat to unbounded so the runner keeps generating until the model
    # naturally stops, instead of truncating at a single token cap. Callers can
    # still force a bounded response with unbounded=False / a max_new_tokens cap.
    if "unbounded" not in out and not kwargs.get("max_new_tokens"):
        out["unbounded"] = True
    return ChatRequest(**out)


def _build_vision_chat_request(kwargs: Dict[str, Any], model_key: str) -> ChatRequest:
    """llama.cpp vision (GGUF + mmproj) rides the chat path.

    Unlike ``_build_chat_request``, an *image* attachment is NOT flattened to
    text (which would raise "text-generation can't consume an image"). Instead
    the image stays on ``ChatRequest.file`` and the llama.cpp runner folds it
    into the latest user turn as an OpenAI ``image_url`` part for the multimodal
    chat handler. Non-image files (docs/text) and imageless turns fall back to
    the normal chat builder unchanged, so a VL model still answers text turns.
    """
    # /ml/vision delivers the image under "image_path"; chat attachments use
    # "file". Accept either (mirrors _build_vision_request) — reading only "file"
    # silently dropped every /ml/vision image to a text-only turn.
    file = kwargs.get("file") or kwargs.get("image_path")
    is_image = bool(file) and derive_media_type(file) == "image"
    if not is_image:
        return _build_chat_request(kwargs, model_key)
    # Build the chat request WITHOUT the image (so it isn't text-flattened),
    # then re-attach it on .file for the runner to pick up (ChatRequest is
    # frozen, so copy with the update).
    kw = {k: v for k, v in kwargs.items() if k not in ("file", "image_path")}
    return _build_chat_request(kw, model_key).model_copy(update={"file": file})


def _build_vision_request(kwargs: Dict[str, Any], model_key: str) -> VisionRequest:
    image_path = kwargs.get("image_path") or kwargs.get("file")
    image_b64 = kwargs.get("image_b64")
    if image_path is None and image_b64 is None:
        raise ValueError(
            "vision request needs 'image_path', 'file', or 'image_b64'; "
            f"got keys: {sorted(kwargs)}"
        )

    out: Dict[str, Any] = {
        "model_key": model_key,
        "request_id": kwargs.get("request_id", make_request_id()),
    }
    # VisionRequest enforces exactly one image source.
    if image_path is not None:
        out["image_path"] = image_path
    else:
        out["image_b64"] = image_b64
    for k in ("prompt", "max_new_tokens", "max_tokens", "pool"):
        if k in kwargs:
            out[k] = kwargs[k]
    return VisionRequest(**out)


def _build_whisper_request(kwargs: Dict[str, Any], model_key: str) -> TranscribeRequest:
    file_path = kwargs.get("audio_path") or kwargs.get("file")
    if file_path is None:
        raise ValueError(
            "whisper request needs 'audio_path' or 'file'; "
            f"got keys: {sorted(kwargs)}"
        )
    out: Dict[str, Any] = {
        "model_key": model_key,
        "file_path": file_path,
        "request_id": kwargs.get("request_id", make_request_id()),
    }
    for k in ("model_size", "language", "capture_frames",
              "min_gap_seconds", "long_segment_seconds", "pool"):
        if k in kwargs:
            out[k] = kwargs[k]
    # 'task' in prompt_kwargs is the dispatch task key, so whisper's own
    # transcribe/translate switch rides separate names.
    whisper_task = kwargs.get("whisper_task")
    if whisper_task is None and kwargs.get("translate"):
        whisper_task = "translate"
    if whisper_task is not None:
        out["task"] = whisper_task
    return TranscribeRequest(**out)


def _build_summarize_request(kwargs: Dict[str, Any], model_key: str) -> "SummarizeRequest":
    text = kwargs.get("text") or kwargs.get("prompt")
    if text is None and kwargs.get("file"):
        text = read_from_file(kwargs["file"])
    if text is None:
        raise ValueError(
            "summarize request needs 'text', 'prompt', or 'file'; "
            f"got keys: {sorted(kwargs)}"
        )

    out: Dict[str, Any] = {
        "model_key": model_key,
        "text": text,
        "request_id": kwargs.get("request_id", make_request_id()),
    }
    for k in (
        "preset", "summary_mode", "input_policy",
        "max_chunk_tokens", "min_length", "max_length",
        "do_sample", "min_input_words",
        "consolidation_min_length", "consolidation_max_length",
        "max_output_words", "pool",
    ):
        if k in kwargs:
            out[k] = kwargs[k]
    return SummarizeRequest(**out)


def _texts_from_kwargs(kwargs: Dict[str, Any]) -> list[str]:
    """Shared text extraction: texts | text | prompt | file -> list[str].

    Used by both embed builders. Returns a list even for single-string
    input so the runner doesn't have to branch.
    """
    raw = kwargs.get("texts") or kwargs.get("text") or kwargs.get("prompt")
    if raw is None and kwargs.get("file"):
        raw = read_from_file(kwargs["file"])
    if raw is None:
        raise ValueError(
            "embed request needs 'texts', 'text', 'prompt', or 'file'; "
            f"got keys: {sorted(kwargs)}"
        )
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list) and all(isinstance(t, str) for t in raw):
        return list(raw)
    raise TypeError(
        f"embed input must be str or list[str], got {type(raw).__name__}"
    )


def _build_embed_request(kwargs: Dict[str, Any], model_key: str) -> EmbedRequest:
    return EmbedRequest(
        model_key=model_key,
        request_id=kwargs.get("request_id", make_request_id()),
        pool=kwargs.get("pool"),
        texts=_texts_from_kwargs(kwargs),
        normalize=kwargs.get("normalize", True),
        batch_size=kwargs.get("batch_size", 32),
    )


def _build_similarity_request(kwargs: Dict[str, Any], model_key: str) -> EmbedRequest:
    """sentence-similarity needs a second set of texts to compare against."""
    other_raw = (
        kwargs.get("other_texts")
        or kwargs.get("other_text")
        or kwargs.get("compare_to")
    )
    if other_raw is None:
        raise ValueError(
            "sentence-similarity needs 'other_texts', 'other_text', or 'compare_to' "
            f"in addition to 'texts'/'text'/'prompt'/'file'; got keys: {sorted(kwargs)}"
        )
    if isinstance(other_raw, str):
        other_texts = [other_raw]
    elif isinstance(other_raw, list) and all(isinstance(t, str) for t in other_raw):
        other_texts = list(other_raw)
    else:
        raise TypeError(
            f"other_texts must be str or list[str], got {type(other_raw).__name__}"
        )

    return EmbedRequest(
        model_key=model_key,
        request_id=kwargs.get("request_id", make_request_id()),
        pool=kwargs.get("pool"),
        texts=_texts_from_kwargs(kwargs),
        other_texts=other_texts,
        normalize=kwargs.get("normalize", True),
        batch_size=kwargs.get("batch_size", 32),
    )


def _build_imagegen_request(kwargs: Dict[str, Any], model_key: str) -> ImageGenRequest:
    prompt = kwargs.get("prompt") or kwargs.get("text")
    if prompt is None and kwargs.get("file"):
        prompt = read_from_file(kwargs["file"])
    if prompt is None:
        raise ValueError(
            "text-to-image request needs 'prompt', 'text', or 'file'; "
            f"got keys: {sorted(kwargs)}"
        )

    out: Dict[str, Any] = {
        "model_key": model_key,
        "prompt": prompt,
        "request_id": kwargs.get("request_id", make_request_id()),
    }
    for k in ("negative_prompt", "width", "height", "num_inference_steps",
              "guidance_scale", "sampler_name", "scheduler", "seed",
              "num_images", "return_b64", "pool"):
        if k in kwargs:
            out[k] = kwargs[k]
    # 'steps' is the colloquial alias clients reach for first.
    if "steps" in kwargs and "num_inference_steps" not in out:
        out["num_inference_steps"] = kwargs["steps"]
    # 'sampler' is the colloquial alias (matches presets.py's field name).
    if "sampler" in kwargs and "sampler_name" not in out:
        out["sampler_name"] = kwargs["sampler"]
    return ImageGenRequest(**out)


def _build_img2img_request(kwargs: Dict[str, Any], model_key: str) -> ImageGenRequest:
    """image-to-image (img2img): text2img PLUS a mandatory init image.

    Mirrors _build_imagegen_request, but ALSO resolves the init image. CRITICAL:
    a worker rematerializes the inlined image under key ``file`` (not
    ``image_path``) — so accept EITHER, exactly like _build_vision_request. Raises
    ValueError with a clear message when no init image is present (img2img has
    nothing to condition on)."""
    prompt = kwargs.get("prompt") or kwargs.get("text")
    if prompt is None and kwargs.get("file"):
        # a text file may carry the prompt; an image file is the init image, not
        # the prompt — only read text/document files as prompt text.
        f = kwargs["file"]
        if derive_media_type(f) in ("text", "document", "code"):
            prompt = read_from_file(f)
    if prompt is None:
        raise ValueError(
            "image-to-image request needs 'prompt' or 'text'; "
            f"got keys: {sorted(kwargs)}"
        )

    image_path = kwargs.get("image_path") or kwargs.get("file")
    if image_path is None:
        raise ValueError(
            "image-to-image request needs an init image via 'image_path' or "
            f"'file'; got keys: {sorted(kwargs)}"
        )

    out: Dict[str, Any] = {
        "model_key": model_key,
        "prompt": prompt,
        "image_path": image_path,
        "request_id": kwargs.get("request_id", make_request_id()),
    }
    for k in ("negative_prompt", "width", "height", "num_inference_steps",
              "guidance_scale", "sampler_name", "scheduler", "seed",
              "num_images", "return_b64", "pool", "strength"):
        if k in kwargs:
            out[k] = kwargs[k]
    # 'steps' is the colloquial alias clients reach for first.
    if "steps" in kwargs and "num_inference_steps" not in out:
        out["num_inference_steps"] = kwargs["steps"]
    # 'sampler' is the colloquial alias (matches presets.py's field name).
    if "sampler" in kwargs and "sampler_name" not in out:
        out["sampler_name"] = kwargs["sampler"]
    return ImageGenRequest(**out)


def _build_keywords_request(kwargs: Dict[str, Any], model_key: str) -> KeywordTaskRequest:
    text = kwargs.get("text") or kwargs.get("prompt")
    if text is None and kwargs.get("file"):
        text = read_from_file(kwargs["file"])
    if text is None:
        raise ValueError(
            "keyword-extraction request needs 'text', 'prompt', or 'file'; "
            f"got keys: {sorted(kwargs)}"
        )

    out: Dict[str, Any] = {
        "model_key": model_key,
        "text": text,
        "request_id": kwargs.get("request_id", make_request_id()),
    }
    for k in ("preset", "refine", "top_n", "diversity", "use_mmr",
              "stop_words", "keyphrase_ngram_range",
              "min_density", "max_density", "min_score", "max_words_per_phrase", "pool"):
        if k in kwargs:
            out[k] = kwargs[k]
    return KeywordTaskRequest(**out)


def _build_vision_analysis_request(kwargs: Dict[str, Any], model_key: str) -> "VisionAnalysisRequest":
    """One builder for the whole vision-analysis family (depth, detection,
    classification, segmentation): input is always one image."""
    image_path = kwargs.get("image_path") or kwargs.get("file")
    image_b64 = kwargs.get("image_b64")
    if image_path is None and image_b64 is None:
        raise ValueError(
            "vision-analysis request needs 'image_path', 'file', or "
            f"'image_b64'; got keys: {sorted(kwargs)}"
        )
    if image_path is not None and derive_media_type(image_path) != "image":
        raise ValueError(
            f"vision-analysis needs an image file; got "
            f"{derive_media_type(image_path)!r} ({os.path.basename(image_path)})"
        )

    out: Dict[str, Any] = {
        "model_key": model_key,
        "request_id": kwargs.get("request_id", make_request_id()),
    }
    if image_path is not None:
        out["image_path"] = image_path
    else:
        out["image_b64"] = image_b64
    for k in ("top_k", "threshold", "candidate_labels", "return_b64", "pool"):
        if k in kwargs:
            out[k] = kwargs[k]
    return VisionAnalysisRequest(**out)


# ---------------------------------------------------------------------------
# Registries — single source of truth.
# ---------------------------------------------------------------------------

MODEL_REQUEST_BUILDERS: Dict[Tuple[str, str], Callable[[Dict[str, Any], str], BaseModel]] = {
    # ComfyUI engine reuses the imagegen request shapes verbatim (same
    # precedent as Img2ImgRunner) — no comfy-specific builders needed.
    ("comfy", "text-to-image"):                       _build_imagegen_request,
    ("comfy", "image-to-image"):                      _build_img2img_request,
    ("transformers", "text-generation"):              _build_chat_request,
    ("gguf",         "text-generation"):              _build_chat_request,
    ("transformers", "image-text-to-text"):           _build_vision_request,
    # GGUF vision rides the chat path: the image stays on ChatRequest.file and
    # the runner attaches it as an image_url part for the multimodal handler.
    ("gguf",         "image-text-to-text"):           _build_vision_chat_request,
    ("transformers", "automatic-speech-recognition"): _build_whisper_request,
    ("transformers", "text-summarization"):                _build_summarize_request,
    ("transformers", "text2text-generation"):         _build_summarize_request,
    ("transformers", "feature-extraction"):           _build_embed_request,
    ("transformers", "sentence-similarity"):          _build_similarity_request,
    ("transformers", "text-to-image"):                _build_imagegen_request,
    ("transformers", "image-to-image"):               _build_img2img_request,
    ("transformers", "keyword-extraction"):           _build_keywords_request,
    ("transformers", "depth-estimation"):             _build_vision_analysis_request,
    ("transformers", "object-detection"):             _build_vision_analysis_request,
    ("transformers", "image-classification"):         _build_vision_analysis_request,
    ("transformers", "image-segmentation"):           _build_vision_analysis_request,
}





