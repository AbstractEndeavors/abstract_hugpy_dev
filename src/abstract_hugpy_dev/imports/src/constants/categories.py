from .imports import *
from .constants import *
# pipeline_tag -> task vocabulary. Every value is HF-derivable.
HF_TASK_TO_TASKS = {
    "text-generation": ["text-generation"],
    "image-text-to-text": ["image-text-to-text", "text-generation"],
    "automatic-speech-recognition": ["automatic-speech-recognition"],
    "text-summarization": ["text-summarization"],
    "text2text-generation": ["text-summarization", "text2text-generation"],
    # KeyBERT rides any sentence-transformers model, so embedding models
    # also serve keyword-extraction.
    "feature-extraction": ["feature-extraction", "sentence-similarity", "keyword-extraction"],
    "sentence-similarity": ["feature-extraction", "sentence-similarity", "keyword-extraction"],
    "text-to-image": ["text-to-image"],
    # NOTE: "image-to-image" is deliberately NOT in the discovery vocabulary.
    # sd-turbo is the ONLY supported img2img model (advertised via its curated
    # tasks list at go-live — see models_config.py step-6). Omitting it here keeps
    # model DISCOVERY from auto-advertising externally-downloaded
    # pipeline_tag=image-to-image models (e.g. flux / Qwen-Image-Edit) as servable
    # img2img before they are vetted. The ("transformers","image-to-image")
    # RUNNER_PAIRS entry below is REQUIRED and stays (it prevents sd-turbo being
    # dropped by derive_model_config_row when step-6 is flipped).
    # Vision-analysis family — HF pipeline tags map 1:1.
    "depth-estimation": ["depth-estimation"],
    "object-detection": ["object-detection"],
    "image-classification": ["image-classification"],
    "image-segmentation": ["image-segmentation"],
}
RUNNER_PAIRS = {
    ("transformers", "text-generation"), ("llama_cpp", "text-generation"),
    ("transformers", "image-text-to-text"), ("llama_cpp", "image-text-to-text"),
    ("transformers", "automatic-speech-recognition"),
    ("transformers", "text-summarization"), ("transformers", "text2text-generation"),
    ("transformers", "feature-extraction"), ("transformers", "sentence-similarity"),
    ("transformers", "text-to-image"), ("transformers", "image-to-image"),
    ("transformers", "keyword-extraction"),
    ("transformers", "depth-estimation"), ("transformers", "object-detection"),
    ("transformers", "image-classification"), ("transformers", "image-segmentation"),
}
MEDIA_DEFAULTS: Dict[str, str] = {
    "document": DEFAULT_CHAT_MODEL,
    "code":     DEFAULT_CHAT_MODEL,
    "text":     DEFAULT_CHAT_MODEL,
    "image":    DEFAULT_VISION_MODEL,
    "audio":    DEFAULT_WHISPER_MODEL,
    "video":    DEFAULT_WHISPER_MODEL,
}

TASK_DEFAULTS: Dict[str, str] = {
    "text-generation":              DEFAULT_CHAT_MODEL,
    "image-text-to-text":           DEFAULT_VISION_MODEL,
    "automatic-speech-recognition": DEFAULT_WHISPER_MODEL,
    "text-summarization":           DEFAULT_SUMMARIZE_MODEL,
    "text2text-generation":         DEFAULT_SUMMARIZE_MODEL,
    "feature-extraction":           DEFAULT_EMBED_MODEL,
    "sentence-similarity":          DEFAULT_EMBED_MODEL,
    "text-to-image":                DEFAULT_IMAGEGEN_MODEL,
    "image-to-image":               DEFAULT_IMAGEGEN_MODEL,
    "keyword-extraction":           DEFAULT_KEYWORDS_MODEL,
    "depth-estimation":             DEFAULT_DEPTH_MODEL,
    "object-detection":             DEFAULT_DETECT_MODEL,
    "image-classification":         DEFAULT_IMG_CLASSIFY_MODEL,
    "image-segmentation":           DEFAULT_SEGMENT_MODEL,
}
