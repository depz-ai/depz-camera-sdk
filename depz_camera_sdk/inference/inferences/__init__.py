"""Concrete :class:`DepthInferenceBase` subclasses, one per file.

Each module here defines exactly one concrete inference — the class
the pipeline holds at runtime. Per-network neural classes subclass
:class:`NeuralStereoInferenceBase` or :class:`NeuralMonoInferenceBase`
(stereo / mono); classical matchers subclass
:class:`ClassicStereoInferenceBase`; rectify-only is its own thing.

Each concrete inference declares its own catalog (:meth:`variants`,
:attr:`LABEL`, :attr:`OUTPUTS_DEPTH`, :attr:`NEURAL`) — the metadata
the UI walks to populate the model picker. ``@register_inference("slug")``
adds the class to the registry; the decorator wraps :meth:`variants`
to stamp the inference slug onto each yielded :class:`InferenceVariant`.
"""

from . import lite_any_stereo  # noqa: F401
from . import depz_stereo  # noqa: F401
from . import lightstereo  # noqa: F401
from . import crestereo_initial  # noqa: F401
from . import crestereo_combined  # noqa: F401
from . import c_fast_foundationstereo  # noqa: F401
from . import depth_anything_v2_small  # noqa: F401
from . import metric3d_v2_vit_small  # noqa: F401
from . import sgbm  # noqa: F401
from . import stereo_bm  # noqa: F401
from . import rectify_only  # noqa: F401

from .lite_any_stereo import LiteAnyStereoInference
from .depz_stereo import DepzStereoInference
from .lightstereo import LightStereoInference
from .crestereo_initial import CREStereoInitialInference
from .crestereo_combined import CREStereoCombinedInference
from .c_fast_foundationstereo import CFastFoundationStereoInference
from .depth_anything_v2_small import DepthAnythingV2SmallInference
from .metric3d_v2_vit_small import Metric3DV2VitSmallInference
from .sgbm import SgbmInference
from .stereo_bm import StereoBMInference
from .rectify_only import RectifyOnlyInference

__all__ = [
    "LiteAnyStereoInference",
    "DepzStereoInference",
    "LightStereoInference",
    "CREStereoInitialInference",
    "CREStereoCombinedInference",
    "CFastFoundationStereoInference",
    "DepthAnythingV2SmallInference",
    "Metric3DV2VitSmallInference",
    "SgbmInference",
    "StereoBMInference",
    "RectifyOnlyInference",
]
