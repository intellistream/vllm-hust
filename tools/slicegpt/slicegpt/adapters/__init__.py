"""Architecture adapters supported by the SliceGPT construction toolkit."""

from .llama_adapter import LlamaModelAdapter
from .qwen2_adapter import Qwen2ModelAdapter

__all__ = ["LlamaModelAdapter", "Qwen2ModelAdapter"]
