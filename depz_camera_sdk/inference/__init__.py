"""Inference algorithms — neural and classical, behind one ABC.

The pipeline holds exactly one :class:`DepthInferenceBase` instance
for its lifetime. Layout:

- :mod:`.base` — every base class (``DepthInferenceBase``,
  ``NeuralStereoInferenceBase``, ``NeuralMonoInferenceBase``,
  ``ClassicStereoInferenceBase``) plus :class:`InferenceVariant` /
  :class:`ModelFile` data types and the download flow.
- :mod:`.registry` — slug → inference class registry,
  ``@register_inference`` decorator, ``cache_root``.
- :mod:`.inferences` — concrete :class:`DepthInferenceBase`
  subclasses, one file per network. Each declares its own catalog
  (``_variants()``, :attr:`LABEL`, :attr:`OUTPUTS_DEPTH`,
  :attr:`NEURAL`).

Adding a new neural network: drop a class subclassing
:class:`NeuralStereoInferenceBase` or :class:`NeuralMonoInferenceBase`
into ``inferences/``, set its input recipe (``BLOB_MEAN`` /
``BLOB_SCALE``) and any output quirk (``OUTPUT_CHANNEL``,
``OUTPUT_NAME``, ``INPUT_SIZE``) as class attributes — override
``preprocess`` / ``postprocess`` only if the defaults do not fit —
declare its variants in ``_variants()``, register with
``@register_inference("slug")``, and :func:`build_inference` picks it
up automatically.
"""

from .base import (
    IMAGENET_BLOB_MEAN,
    IMAGENET_BLOB_SCALE,
    DepthInferenceBase,
    InferenceVariant,
    Control,
    ModelFile,
    NeuralMonoInferenceBase,
    NeuralStereoInferenceBase,
    build_inference,
    onnx_variant,
    stderr_progress,
)
from .inferences import (
    CREStereoCombinedInference,
    CREStereoInitialInference,
    DepthAnythingV2SmallInference,
    CFastFoundationStereoInference,
    DepzStereoInference,
    LightStereoInference,
    LiteAnyStereoInference,
    LiteAnyStereoMInference,
    Metric3DV2VitSmallInference,
    RectifyOnlyInference,
    SgbmInference,
    StereoBMInference,
)
from .registry import (
    available_inferences,
    cache_root,
    get_inference,
    register_inference,
)


__all__ = [
    # ABC + descriptor
    "DepthInferenceBase", "Control",
    # Intermediate bases
    "NeuralStereoInferenceBase", "NeuralMonoInferenceBase",
    # Concrete inferences
    "LiteAnyStereoInference",
    "LiteAnyStereoMInference",
    "DepzStereoInference",
    "LightStereoInference",
    "CREStereoInitialInference",
    "CREStereoCombinedInference",
    "CFastFoundationStereoInference",
    "DepthAnythingV2SmallInference",
    "Metric3DV2VitSmallInference",
    "SgbmInference", "StereoBMInference",
    "RectifyOnlyInference",
    # Variant / catalog
    "InferenceVariant", "ModelFile", "onnx_variant",
    "available_inferences", "get_inference", "register_inference",
    "cache_root",
    # Input recipes
    "IMAGENET_BLOB_MEAN", "IMAGENET_BLOB_SCALE",
    # Variant-driven factory
    "build_inference", "stderr_progress",
]
