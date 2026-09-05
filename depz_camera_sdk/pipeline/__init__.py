"""DepthPipeline + everything that lives inside its lifecycle.

Public API: :class:`DepthPipeline` plus the value types it produces
(:class:`FramePacket`, :class:`PipelineMetrics`,
:class:`RecordingStatus`, :class:`RecordingResult`).

Internal helpers (factory, postprocessor, metrics counters,
recording_controller, recorder) are reachable as submodules but
aren't re-exported here — application code shouldn't need them.
"""

from .depth_pipeline import DepthPipeline
from .types import (
    PipelineMetrics,
    RecordingResult,
    RecordingStatus,
    FramePacket,
)

__all__ = [
    "DepthPipeline",
    "FramePacket",
    "PipelineMetrics",
    "RecordingStatus",
    "RecordingResult",
]
