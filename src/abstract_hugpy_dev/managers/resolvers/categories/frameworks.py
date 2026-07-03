from .imports import *
FRAMEWORK_RUNNERS: Dict[Tuple[str, str], Type[Runner]] = {
    ("transformers", "text-generation"):              DeepCoderChatRunner,
    ("llama_cpp",    "text-generation"):              LlamaCppChatRunner,
    ("transformers", "image-text-to-text"):           VisionRunner,
    # GGUF vision (e.g. Qwen2.5-VL-*-GGUF): the gguf chat runner takes the same
    # ChatRequest and forwards image_url content in messages to llama.cpp's
    # chat-completion path — no separate VisionRunner shape.
    ("llama_cpp",    "image-text-to-text"):           LlamaCppChatRunner,
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

