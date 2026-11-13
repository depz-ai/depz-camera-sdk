"""Worker-queue command objects + the inflight-slot dataclass.

Every public DepthPipeline method that mutates worker-owned state
(filters, inference controls, calibration, streams, photo mode, seek)
enqueues one of these. The pipeline's worker thread drains the queue
between frames and dispatches each command to a ``_handle_*`` method.

Pulled out of ``depth_pipeline.py`` so the command vocabulary is
visible at a glance — adding a new RPC mutation is "add a dataclass
here, add a handler in the pipeline, add the dispatch entry". The
pipeline file stays focused on the worker loop + handlers; the command
shapes themselves are inert data.

All names are leading-underscore — these are pipeline-internal, not
part of the public API. Direct callers should go through the
:class:`DepthPipeline` methods that build and enqueue them.
"""

import threading
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..engines import InferenceFuture


# ── Command object hierarchy (enqueued from any thread) ────────────────


@dataclass
class _Command:
    event: threading.Event = field(default_factory=threading.Event)
    error: Optional[BaseException] = None


@dataclass
class _SetCalibrationControlCommand(_Command):
    """Generic calibration-control update. Replaces the old per-param
    ``_SetRectifyAlphaCommand``. Concrete :class:`Calibration` impl
    decides which control names it understands (e.g. opencv_basic accepts
    ``'rectify_alpha'``)."""
    name: str = ""
    value: Any = None
    applied_value: Any = None


@dataclass
class _UpdateCalibrationCommand(_Command):
    """Online-recalibrator hook. ``patch`` is forwarded verbatim to
    :meth:`Calibration.apply_update` — opaque to the SDK."""
    patch: dict = field(default_factory=dict)


@dataclass
class _SetFilterCommand(_Command):
    filter_slug: str = ""
    name: str = ""
    value: Any = None
    applied_value: Any = None


@dataclass
class _SetInferenceControlCommand(_Command):
    """Mutate one :class:`DepthInferenceBase` control (e.g. SGBM
    block_size). Routed through the worker so backends that need to
    rebuild internal state (cv2.StereoSGBM matcher) do so off the API
    thread."""
    name: str = ""
    value: Any = None
    applied_value: Any = None


@dataclass
class _SetStreamEnabledCommand(_Command):
    name: str = ""
    enabled: bool = True
    cascade: bool = False
    applied: bool = False
    reason: str = ""
    # Names that were also flipped because of cascade (excluding `name`
    # itself). Empty when cascade=False or no extra streams matched.
    cascaded_names: tuple = ()


@dataclass
class _SeekCommand(_Command):
    target_us: int = 0


@dataclass
class _StepReplayCommand(_Command):
    """Step exactly one frame forward / backward in MKV replay.
    Distinct from :class:`_SeekCommand` because it doesn't reset
    stateful filters or ``frame_number`` — temporal continuity is
    preserved across a single-frame step."""
    direction: str = "forward"      # "forward" | "backward"


@dataclass
class _TriggerSnapshotCommand(_Command):
    """Arm a one-shot inference for the next frame. Idempotent —
    setting ``_snapshot_pending = True`` (boolean, not a counter) so
    repeated triggers before the snapshot fires don't queue extra
    work. No-op outside photo mode (inference runs every frame)."""


@dataclass
class _SetRecalibrationControlCommand(_Command):
    """Mutate one :class:`RecalibrationController` control (e.g. ``ok_px``
    or the ``enabled`` on/off). Routed through the worker so the
    controller's ``control_set`` side effects (rebuilding the health
    monitor on a ``window``/``ok_px`` change) run off the API thread.
    Mirrors :class:`_SetInferenceControlCommand`.
    No-op target (``self._recal_controller is None``) raises in
    :meth:`DepthPipeline.set_recalibration_control` before enqueuing."""
    name: str = ""
    value: Any = None
    applied_value: Any = None


@dataclass
class _SetPipelineControlCommand(_Command):
    """Generic pipeline-control mutation. Worker resolves ``name`` against
    :meth:`DepthPipeline.controls`, coerces via the descriptor, stores,
    and dispatches the per-control side effect. Mirrors the
    filter / inference / calibration command pattern.

    Carries ``photo_mode`` and ``output_drop_policy`` today; future
    controls (rate-limits, worker-level toggles, ...) plug into the same
    path without growing more command types."""
    name: str = ""
    value: Any = None
    applied_value: Any = None       # filled by worker on success


# ── Inflight tracking ─────────────────────────────────────────────────


@dataclass(slots=True)
class _InflightSlot:
    """One in-flight inference. Per-frame, lives until its future
    completes and is post-processed into a FramePacket."""
    future: InferenceFuture
    frame_number: int                # pipeline-counter at submit time
    left: Optional[np.ndarray]       # raw left, if "raw" stream enabled
    right: Optional[np.ndarray]
    rect_l: Optional[np.ndarray]     # rectified left, if "rectified" enabled
    rect_r: Optional[np.ndarray]
    capture_pts_us: int
    submit_ts_ns: int
    source_frame_idx: Optional[int] = None
    is_snapshot: bool = False        # photo-mode snapshot trigger
