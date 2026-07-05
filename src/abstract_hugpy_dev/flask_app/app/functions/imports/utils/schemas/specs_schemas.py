from .imports import *

class Runtime(str, Enum):
    transformers = "transformers"
    gguf = "gguf"                 # HF Hub's library tag for GGUF repos
    dataset = "dataset"
    unknown = "unknown"


