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
  (:meth:`variants`, :attr:`LABEL`, :attr:`OUTPUTS_DEPTH`,
  :attr:`NEURAL`).

Adding a new neural network: drop a class subclassing
:class:`NeuralStereoInferenceBase` or :class:`NeuralMonoInferenceBase`
into ``inferences/`` with its own ``preprocess`` / ``postprocess``,
register with ``@register_inference("slug")``, declare its variants
on the class, and :func:`build_inference` picks it up automatically.
"""

from .base import (
    DepthInferenceBase,
    InferenceVariant,
    Control,
    ModelFile,
    NeuralMonoInferenceBase,
    NeuralStereoInferenceBase,
    build_inference,
)
from .inferences import (
    CREStereoCombinedInference,
    CREStereoInitialInference,
    DepthAnythingV2SmallInference,
    CFastFoundationStereoInference,
    DepzStereoInference,
    LightStereoInference,
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
    "InferenceVariant", "ModelFile",
    "available_inferences", "get_inference", "register_inference",
    "cache_root",
    # Variant-driven factory
    "build_inference",
]
