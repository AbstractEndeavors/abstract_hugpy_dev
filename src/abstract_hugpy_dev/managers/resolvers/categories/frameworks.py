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
    ("transformers", "keyword-extraction"):           KeywordRunner,
}

# Derived from FRAMEWORK_RUNNERS so it can't drift.
KNOWN_TASKS_REGISTRY: frozenset[str] = frozenset(task for _, task in FRAMEWORK_RUNNERS.keys())

