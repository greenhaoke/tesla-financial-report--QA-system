# core/embedder/__init__.py
from .factory import get_embedder
from .base_embedder import BaseEmbedder

__all__ = ["get_embedder", "BaseEmbedder"]
