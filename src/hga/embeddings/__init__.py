"""Embeddings package — interface + factory."""

from __future__ import annotations

from hga.config import Settings, get_settings
from hga.embeddings.base import Embedder
from hga.embeddings.local_embedder import LocalTextEmbedder
from hga.embeddings.voyage import VoyageMultimodalEmbedder, VoyageTextEmbedder

__all__ = ["Embedder", "get_text_embedder", "get_multimodal_embedder"]


def get_text_embedder(settings: Settings | None = None) -> Embedder:
    """Return the text embedder for the active config profile."""
    s = settings or get_settings()
    if s.profile == "local_gpu":
        return LocalTextEmbedder(model=s.local_embed_model)
    if s.profile == "local":
        return VoyageTextEmbedder(model=s.text_embed_model)
    raise NotImplementedError(f"Embeddings profile {s.profile!r} not yet implemented")


def get_multimodal_embedder(settings: Settings | None = None) -> Embedder:
    """Return the multimodal embedder for the active config profile.

    For local_gpu, falls back to the text embedder (embed captions only).
    """
    s = settings or get_settings()
    if s.profile == "local_gpu":
        return get_text_embedder(s)
    if s.profile == "local":
        return VoyageMultimodalEmbedder(model=s.multimodal_embed_model)
    raise NotImplementedError(f"Embeddings profile {s.profile!r} not yet implemented")
