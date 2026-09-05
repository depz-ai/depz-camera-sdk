"""Inference registry + cache-root helper.

:data:`_INFERENCE_REGISTRY` — slug → concrete :class:`DepthInferenceBase`
subclass. Populated by ``@register_inference("slug")`` decorators on
the per-network modules under ``inferences/``. Each inference declares
its own catalog (:meth:`variants`, :attr:`LABEL`, :attr:`OUTPUTS_DEPTH`,
:attr:`NEURAL`) as classmethods / class attributes.

The decorator wraps :meth:`variants` to stamp ``_inference_slug`` on
every yielded :class:`InferenceVariant` so the variant carries a
back-pointer to its inference class — the cache directory layout,
download flow, and :func:`build_inference` all need this slug.

Re-registration of the *same class object* is a no-op (notebook
hot-reload); re-registering a *different* class under an
already-claimed slug raises so the operator notices the collision.
"""

import dataclasses
from pathlib import Path
from typing import Type

from .base import DepthInferenceBase, InferenceVariant


_INFERENCE_REGISTRY: dict[str, Type[DepthInferenceBase]] = {}


def register_inference(slug: str):
    """Decorator: register a concrete :class:`DepthInferenceBase`
    subclass under ``slug``. Stamps :attr:`SLUG`, wraps
    :meth:`variants` to bind the slug onto each yielded
    :class:`InferenceVariant`, and adds the class to
    :data:`_INFERENCE_REGISTRY`."""
    def _wrap(cls: Type[DepthInferenceBase]) -> Type[DepthInferenceBase]:
        if not issubclass(cls, DepthInferenceBase):
            raise TypeError(
                f"{cls.__name__} must inherit from DepthInferenceBase "
                f"to register")
        cls.SLUG = slug

        # Wrap variants() to stamp ``_inference_slug`` on every
        # yielded InferenceVariant. Subclasses construct
        # ``InferenceVariant(slug=..., label=...)`` without thinking
        # about which inference they belong to; the registry — which
        # by definition knows the slug — fills it in.
        _original_variants = cls.variants.__func__

        def _bound_variants(cls_) -> tuple[InferenceVariant, ...]:
            return tuple(
                v if v._inference_slug == slug
                else dataclasses.replace(v, _inference_slug=slug)
                for v in _original_variants(cls_)
            )
        cls.variants = classmethod(_bound_variants)

        existing = _INFERENCE_REGISTRY.get(slug)
        if existing is not None and existing is not cls:
            raise RuntimeError(
                f"inference {slug!r} already registered to "
                f"{existing.__module__}.{existing.__name__}")
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
    """Root directory for downloaded variant cache. Sub-directory of
    the SDK-wide cache root (:func:`depz_camera_sdk.get_cache_dir`), so
    every consumer benefits from the same redirect. Created lazily by
    :meth:`InferenceVariant.resolve_files` when the first variant lands."""
    from ..cache_dir import get_cache_dir
    return get_cache_dir() / "models"
