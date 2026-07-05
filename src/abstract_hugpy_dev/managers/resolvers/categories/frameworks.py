from .imports import *
FRAMEWORK_RUNNERS: Dict[Tuple[str, str], Type[Runner]] = {
    ("transformers", "text-generation"):              DeepCoderChatRunner,
    ("gguf",         "text-generation"):              LlamaCppChatRunner,
    ("transformers", "image-text-to-text"):           VisionRunner,
    # GGUF vision (e.g. Qwen2.5-VL-*-GGUF): the gguf chat runner takes the same
    # ChatRequest and forwards image_url content in messages to llama.cpp's
    # chat-completion path — no separate VisionRunner shape.
    ("gguf",         "image-text-to-text"):           LlamaCppChatRunner,
    ("transformers", "automatic-speech-recognition"): WhisperRunner,
    ("transformers", "text-summarization"):                SummarizeRunner,
    ("transformers", "text2text-generation"):         SummarizeRunner,
    ("transformers", "feature-extraction"):           FeatureExtractionRunner,
    ("transformers", "sentence-similarity"):          FeatureExtractionRunner,
    ("transformers", "text-to-image"):                ImageGenRunner,
    # img2img sibling of text-to-image. INERT until a model advertises the task
    # (sd-turbo's advertisement flip is HELD — see models_config.py). Registering
    # the pair makes "image-to-image" a member of KNOWN_TASKS_REGISTRY and lets
    # validate_registry accept the flip when it goes live.
    ("transformers", "image-to-image"):               Img2ImgRunner,
    # ComfyUI engine (slice B): a comfy row's `filename` names a checkpoint in
    # the WORKER's own ComfyUI install; the runner drives its local :8188 with
    # vanilla-node templates and reuses the imagegen request/result, so
    # delegation + the b64 artifact seam work unchanged.
    ("comfy", "text-to-image"):                       ComfyRunner,
    ("comfy", "image-to-image"):                      ComfyRunner,
    ("transformers", "keyword-extraction"):           KeywordRunner,
    # Vision-analysis family — ONE generic transformers-pipeline runner, a
    # subclass per task (see managers/vision_analysis). Adding the next HF
    # image task = a two-line subclass + a row here and in builders.
    ("transformers", "depth-estimation"):             DepthEstimationRunner,
    ("transformers", "object-detection"):             ObjectDetectionRunner,
    ("transformers", "image-classification"):         ImageClassificationRunner,
    ("transformers", "image-segmentation"):           ImageSegmentationRunner,
}

# Derived from FRAMEWORK_RUNNERS so it can't drift.
KNOWN_TASKS_REGISTRY: frozenset[str] = frozenset(task for _, task in FRAMEWORK_RUNNERS.keys())

