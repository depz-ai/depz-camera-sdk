"""Stereo inference engines (OpenVINO + ONNX-multi-EP) with unified async API."""

from .base import EventFuture, InferenceEngine, InferenceFuture
from .registry import (
    ENGINE_CLASSES, available_engines, class_rank, classify_engine,
    list_engines, make_engine, parse_engine_slug, priority_rank,
    recommend_engine,
)

__all__ = [
    "InferenceEngine", "InferenceFuture", "EventFuture",
    "ENGINE_CLASSES", "available_engines", "class_rank",
    "classify_engine", "list_engines", "make_engine",
    "parse_engine_slug", "priority_rank", "recommend_engine",
]
