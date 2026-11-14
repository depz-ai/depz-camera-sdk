"""Which model runs on which NVIDIA engine.

An NVIDIA GPU can run a model two ways: ONNX Runtime's CUDA provider
(``onnx-cuda`` — runs everything, exact FP32) or TensorRT used directly
(``tensorrt-fp16`` / ``tensorrt-fp32`` — three to four times less GPU
time per frame, so higher frame rates, a cooler GPU and less power).
TensorRT is not safe to assume: one model cannot be read by it, another
loses accuracy in half precision. So every model family has an explicit,
measured entry here, and :func:`engine_for_model` turns "this NVIDIA GPU"
into the fastest engine that keeps the model's accuracy.

This module is *policy* and ships as plain ``.py`` (like ``registry.py``)
so an entry can be changed in the field without a rebuild.

Changing or adding a model:

1. run ``tools/tensorrt/verify_models.py`` on a machine with an NVIDIA
   GPU and TensorRT — it compares each entry with ``onnx-cuda``;
2. put the result in ``tools/tensorrt/results/`` and the entry below.

``tests/test_model_policy.py`` fails while a catalog family has no entry.
"""

import os
from typing import Iterable, Optional

# Inference family → the TensorRT precision it is verified for, or None
# when it must stay on ``onnx-cuda``. A ``"family/variant"`` key overrides
# its family for one variant.
#
# Evidence (RTX 5070 Ti, TensorRT 10.16, against onnx-cuda), files under
# tools/tensorrt/results/:
#   precisions_rtx5070ti_trt10.16_loaded.jsonl   Middlebury, ground truth
#   families_tensorrt_fp16_rtx5070ti.jsonl       synthetic pair, all families
TENSORRT_PRECISION: "dict[str, Optional[str]]" = {
    # 0.015 px median difference, same ground-truth error; 15 → 4 ms.
    "lite-any-stereo": "fp16",
    # 0.014 px; 20 → 6 ms.
    "lite-any-stereo-m": "fp16",
    # FP16 with the layers around its 1e-12 epsilons kept in FP32 (the
    # engine does that itself): 0.013 px; 230 → ~74 ms at 640×384.
    "c-fast-foundationstereo": "fp16",
    # 0.004 px; 6.6 → 2.5 ms.
    "lightstereo": "fp16",
    # 0.0002 px; 117 → 25 ms.
    "crestereo-initial": "fp16",
    "crestereo-combined": "fp16",
    # FP16 is ~6 % off the FP32 depth (not NaN, so nothing catches it).
    # FP32 on TensorRT matches to 0.0006 and is 1.3× faster (34 → 26 ms).
    "metric3d-v2-vit-small": "fp32",
    # TensorRT cannot read the file: it contains ONNX Runtime's own
    # MultiHeadAttention operator. Needs a re-export first.
    "depth-anything-v2-small": None,
    # Not verified on TensorRT — left on CUDA until it is.
    "depz-stereo": None,
}

# The engines that are "this NVIDIA GPU". Only a request for one of them
# is ever rewritten; a request for the CPU or an Intel device is final.
NVIDIA_ENGINES = frozenset({"onnx-cuda", "tensorrt-fp16", "tensorrt-fp32"})

# Escape hatch: DEPZ_DISABLE_TENSORRT=1 keeps everything on onnx-cuda.
_DISABLE_ENV = "DEPZ_DISABLE_TENSORRT"


def tensorrt_precision(inference_slug: str,
                       variant_slug: "Optional[str]" = None) -> Optional[str]:
    """The precision ``inference_slug`` is verified for on TensorRT
    (``"fp16"`` / ``"fp32"``), or None: not verified, stay on CUDA.
    Families missing from the table (a user's own model) are None."""
    if variant_slug is not None:
        key = f"{inference_slug}/{variant_slug}"
        if key in TENSORRT_PRECISION:
            return TENSORRT_PRECISION[key]
    return TENSORRT_PRECISION.get(inference_slug)


def engine_for_model(inference_slug: str, variant_slug: "Optional[str]",
                     engine_slug: "Optional[str]",
                     engines: "Optional[Iterable[str]]" = None,
                     ) -> "Optional[str]":
    """The engine that should run this model when ``engine_slug`` was
    asked for.

    Requests for anything but an NVIDIA engine come back unchanged. For
    an NVIDIA engine the answer is ``tensorrt-<precision>`` when the
    model is verified for it and that engine is installed, else
    ``onnx-cuda`` when that is installed, else the request itself (a
    Jetson has no ``onnx-cuda`` to fall back to).

    ``engines`` — the loadable engine slugs, hidden ones included;
    defaults to ``list_engines("detected")``.
    """
    if engine_slug not in NVIDIA_ENGINES:
        return engine_slug
    if engines is None:
        from .registry import list_engines
        engines = list_engines("detected")
    have = set(engines)
    precision = tensorrt_precision(inference_slug, variant_slug)
    if os.environ.get(_DISABLE_ENV) == "1":
        precision = None
    if precision is not None and f"tensorrt-{precision}" in have:
        return f"tensorrt-{precision}"
    if "onnx-cuda" in have:
        return "onnx-cuda"
    return engine_slug
