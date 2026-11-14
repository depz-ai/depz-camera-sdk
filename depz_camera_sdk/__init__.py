"""DEPZ Camera SDK — public API.

Capture from DEPZ stereo / mono cameras (or replay an ``.mkv``
recording), rectify with the unit's calibration, run a depth model and
get depth maps and point clouds out::

    from depz_camera_sdk import DepthPipeline

    for packet in DepthPipeline():      # first camera, default model
        depth = packet.depth            # HxW float32, metres

The pieces, each importable from here:

    DepthPipeline                 source → rectify → inference → FramePacket
    CameraSource / MkvSource      live camera / recording replay
    build_inference(slug, variant)            a ready DepthInferenceBase
    available_inferences() / get_inference()  the model catalog
    available_engines() / recommend_engine()  inference engines on this host
    list_connected_depz_cameras()             attached cameras + calibration
    Calibration, filters, typed Control descriptors
    get_cache_dir() / set_cache_dir()         where models and caches live

``docs/guide.md`` walks through all of it; ``docs/api.md`` is the
generated reference.
"""

# Flattened multi-module extensions: must be wired BEFORE any subpackage
# import so member modules resolve from their fat .so when one is present
# (no-op in pure-python checkouts). See _flatten.py.
from . import _flatten as _flatten_mod
_flatten_mod.install()
del _flatten_mod

from . import platform as _platform  # triggers env-var + CUDA setup at import time
del _platform


from ._controls import (
    BoolControl,
    Control,
    FloatControl,
    IntControl,
    StrControl,
)
from .cache_dir import get_cache_dir, set_cache_dir
from .logging_setup import configure_logging, enable_default_logging
from .calibration import (
    Calibration,
    CanvasLayout,
    OpenCVBasicCalibration,
    OpenCVFisheyeCalibration,
)
from .cameras import (
    DepzCameraInfo,
    fetch_camera_info_json,
    get_first_connected_depz_camera,
    get_connected_depz_camera_by_serial,
    list_connected_depz_cameras,
)
from .engines import (
    available_engines,
    list_engines,
    recommend_engine,
)
from .inference import (
    InferenceVariant,
    ModelFile,
    NeuralMonoInferenceBase,
    NeuralStereoInferenceBase,
    RectifyOnlyInference,
    SgbmInference,
    StereoBMInference,
    DepthInferenceBase,
    available_inferences,
    build_inference,
    get_inference,
    register_inference,
    stderr_progress,
)
from .filters import (
    BoundingBoxFilter,
    DepthFilter,
    DisparityFilter,
    EdgeFilter,
    Filter,
    PointCloudFilter,
    SpeckleFilter,
    StatisticalOutlierFilter,
    TemporalEmaFilter,
    ZClipFilter,
)
from .pipeline import (
    PipelineMetrics,
    RecordingResult,
    RecordingStatus,
    FramePacket,
    DepthPipeline,
)
from .sources import CameraSource, FrameSource, MkvSource
from .sources.types import CameraInfo, RawFrame
from .cameras.camera_spec import CameraControl, CaptureMode

__all__ = [
    # Pipeline
    "DepthPipeline",
    # Sources
    "FrameSource", "CameraSource", "MkvSource",
    # Inference engines (slugs only — InferenceEngine + make_engine
    # are still importable for advanced users / tests, just not in __all__)
    "available_engines", "list_engines", "recommend_engine",
    # DepthInferenceBase — unified ABC for ML + classical disparity solvers.
    "DepthInferenceBase",
    "NeuralStereoInferenceBase", "NeuralMonoInferenceBase",
    "SgbmInference", "StereoBMInference", "RectifyOnlyInference",
    "build_inference", "stderr_progress",
    # Model catalog + per-network inference plugin API (the concrete
    # network classes live in depz_camera_sdk.inference)
    "available_inferences", "get_inference", "register_inference",
    "InferenceVariant", "ModelFile",
    # Control descriptors — shared by inferences, filters, calibration
    # and the pipeline
    "Control", "IntControl", "FloatControl", "BoolControl", "StrControl",
    # Filters (subclass DisparityFilter/DepthFilter/PointCloudFilter
    # + append to default_filter_chain to add).
    "Filter",
    "DisparityFilter", "DepthFilter", "PointCloudFilter",
    "SpeckleFilter",
    "ZClipFilter", "EdgeFilter", "TemporalEmaFilter",
    "BoundingBoxFilter", "StatisticalOutlierFilter",
    # Calibration
    "Calibration", "CanvasLayout",
    "OpenCVBasicCalibration", "OpenCVFisheyeCalibration",
    "fetch_camera_info_json",
    # SDK cache root (engines + apps subdivide by purpose)
    "get_cache_dir", "set_cache_dir",
    # Logging bootstrap (opt-in; nothing configured at import time)
    "configure_logging", "enable_default_logging",
    # Camera discovery
    "DepzCameraInfo", "list_connected_depz_cameras",
    "get_first_connected_depz_camera",
    "get_connected_depz_camera_by_serial",
    # Types
    "CameraInfo", "CaptureMode", "RawFrame", "FramePacket",
    "RecordingStatus", "RecordingResult",
    "PipelineMetrics", "CameraControl",
]
