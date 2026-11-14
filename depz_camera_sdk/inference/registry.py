"""Inference registry + cache-root helper.

:data:`_INFERENCE_REGISTRY` — slug → concrete :class:`DepthInferenceBase`
subclass. Populated by ``@register_inference("slug")`` decorators on
the per-network modules under ``inferences/``. Each inference declares
its own catalog (:meth:`~DepthInferenceBase._variants`, :attr:`LABEL`,
:attr:`OUTPUTS_DEPTH`, :attr:`NEURAL`) as classmethods / class
attributes.

Re-registration of the *same class object* is a no-op (notebook
hot-reload); re-registering a *different* class under an
already-claimed slug raises so the operator notices the collision.
"""

from pathlib import Path
from typing import Type

from .base import DepthInferenceBase


_INFERENCE_REGISTRY: dict[str, Type[DepthInferenceBase]] = {}


def register_inference(slug: str):
    """Decorator: register a concrete :class:`DepthInferenceBase`
    subclass under ``slug`` and set its :attr:`SLUG`. Nothing is
    changed on the class if registration is refused."""
    def _wrap(cls: Type[DepthInferenceBase]) -> Type[DepthInferenceBase]:
        if not issubclass(cls, DepthInferenceBase):
            raise TypeError(
                f"{cls.__name__} must inherit from DepthInferenceBase "
                f"to register")
        if "variants" in vars(cls):
            raise TypeError(
                f"{cls.__name__} overrides variants(); declare the "
                f"catalog in _variants() instead — variants() is the "
                f"base-class method that binds the inference slug")
        existing = _INFERENCE_REGISTRY.get(slug)
        if existing is not None and existing is not cls:
            raise RuntimeError(
                f"inference {slug!r} already registered to "
                f"{existing.__module__}.{existing.__name__}")
        cls.SLUG = slug
        _INFERENCE_REGISTRY[slug] = cls
        return cls
    return _wrap


def available_inferences() -> tuple[Type[DepthInferenceBase], ...]:
    """Every registered inference class, sorted by slug for
    deterministic UI listing."""
    return tuple(_INFERENCE_REGISTRY[s] for s in sorted(_INFERENCE_REGISTRY))


def get_inference(slug: str) -> Type[DepthInferenceBase]:
    """Look up an inference class by slug. Raises ``KeyError`` with a
    friendly message listing the registered slugs if the slug is
    unknown."""
    inf = _INFERENCE_REGISTRY.get(slug)
    if inf is None:
        raise KeyError(
            f"unknown inference {slug!r}; available: "
            f"{tuple(sorted(_INFERENCE_REGISTRY))}")
    return inf


# ──────────────────────── Cache root ─────────────────────────────────


def cache_root() -> Path:
    """Root directory of the downloaded-model cache: ``models/`` under
    the SDK-wide cache root (:func:`depz_camera_sdk.get_cache_dir`), so
    one :func:`depz_camera_sdk.set_cache_dir` call moves everything.
    Created lazily by :meth:`InferenceVariant.resolve_files` when the
    first variant lands."""
    from ..cache_dir import get_cache_dir
    return get_cache_dir() / "models"
