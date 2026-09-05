"""DEPZ Stereo SDK — public API.

See docs/refactor_plan.md for the architectural overview.

Discovery helpers (use these to wire up a fresh pipeline):

    list_connected_depz_cameras() → list[DepzCameraInfo]   (Varius/Minima/Clarity only)
    get_first_connected_depz_camera()    → DepzCameraInfo        (first connected; raises if none)
    get_connected_depz_camera_by_serial(s) → DepzCameraInfo | None  (single-fetch lookup)
    available_engines()           → list[str] (engine slugs, sorted by priority)
    list_engines(scope=...)       → list[str] ("available"/"detected"/"full"/"hidden")
    recommend_engine()            → str | None (best slug for this host)
"""

from . import platform as _platform  # triggers env-var + CUDA setup at import time
del _platform


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
    Control,
    NeuralMonoInferenceBase,
    NeuralStereoInferenceBase,
    RectifyOnlyInference,
    SgbmInference,
    StereoBMInference,
    DepthInferenceBase,
    available_inferences,
    build_inference,
    register_inference,
)
from .filters import (
    BoundingBoxFilter,
    DepthFilter,
    DisparityFilter,
    EdgeFilter,
    Filter,
    Control,
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
    # Concrete implementations + control descriptor.
    "DepthInferenceBase", "Control",
    "NeuralStereoInferenceBase", "NeuralMonoInferenceBase",
    "SgbmInference", "StereoBMInference", "RectifyOnlyInference",
    "build_inference",
    # Per-network inference plugin API
    "available_inferences", "register_inference",
    # Filters (subclass DisparityFilter/DepthFilter/PointCloudFilter
    # + append to default_filter_chain to add). `Control` is already listed
    # above under inference — it's the shared control descriptor.
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
