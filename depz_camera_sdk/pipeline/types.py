"""Pipeline-side value types — what :class:`DepthPipeline` produces.

:class:`FramePacket` is the per-frame output of :meth:`DepthPipeline.get`.
:class:`RecordingStatus` / :class:`RecordingResult` are the
recording-subsystem status / final-summary types. :class:`PipelineMetrics`
is a snapshot of the live performance counters, returned by
:attr:`DepthPipeline.metrics`.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FramePacket:
    """One processed stereo frame — what `pipeline.get()` returns.

    All array fields are optional and may be None when their stream has
    been disabled via `set_stream_enabled`. Shapes/dtypes are documented
    on each field.
    """
    frame_number: int
    """Pipeline-counter (resets to 0 on `seek()`). Distinct from
    `source_frame_idx`."""

    pts_us: int
    """Mirrors `RawFrame.pts_us` — microseconds since stream start,
    source-local. Use `pipeline.pts_to_monotonic_ns(pts_us)` for
    absolute time."""

    source_frame_idx: Optional[int] = None
    """Mirrors `RawFrame.source_frame_idx` — recording-relative index
    on replay; `None` on live."""

    left: Optional[np.ndarray] = None
    """Raw left eye, HxWx3 uint8 BGR. `None` unless `enable_raw=True`
    (or toggled via `set_stream_enabled`)."""

    right: Optional[np.ndarray] = None
    """Raw right eye, HxWx3 uint8 BGR. Gated by `enable_raw`."""

    rectified_left: Optional[np.ndarray] = None
    """Rectified left eye on inference canvas, h1×w1×3 uint8 BGR.
    Gated by `enable_rectified`."""

    rectified_right: Optional[np.ndarray] = None
    """Rectified right eye on inference canvas, h1×w1×3 uint8 BGR.
    Gated by `enable_rectified`."""

    disparity: Optional[np.ndarray] = None
    """Disparity map in canvas pixels, h1×w1 float32. Off-by-default
    in packet exposure; flip via
    `set_stream_enabled("disparity", True)`."""

    depth: Optional[np.ndarray] = None
    """Metric depth, h1×w1 float32 meters. Default-on (the typical
    'I want stereo output' field)."""

    point_cloud: Optional[np.ndarray] = None
    """Back-projected XYZ, h1×w1×3 float32 meters (camera frame).
    Heaviest stage — gated by `enable_cloud`."""

    is_snapshot: bool = False
    """True iff this packet carries inference results produced by a
    photo-mode snapshot trigger (see
    :meth:`DepthPipeline.trigger_snapshot`). Always False outside
    photo mode and on photo-mode bypass packets. Used by
    :meth:`DepthPipeline.get_snapshot` to filter out the live-preview
    stream while waiting for the result of a Snap."""


@dataclass(frozen=True, slots=True)
class RecordingStatus:
    is_recording: bool
    """True between `start_recording()` and `stop_recording()`."""

    path: Optional[str]
    """Active MKV path; `None` when idle."""

    frames_written: int
    """Frames already muxed (live counter)."""

    bytes_written: int
    """Bytes already on disk (live counter)."""

    dropped: int
    """Live drop counter — frames the recorder couldn't keep up with."""

    duration_s: float
    """Wall-clock seconds since `start_recording()`."""

    is_paused: bool = False
    """True while recording is paused (frames not being written) but the
    output file is still open — a subsequent ``resume_recording()``
    continues the same recording."""

    error: Optional[str] = None
    """Writer-crash description (``"Type: message"``) when the recorder's
    writer thread died mid-recording — the on-disk file is likely
    incomplete. ``None`` while healthy. UI surfaces this as a
    "recording failed" notice instead of a silently frozen counter."""


@dataclass(frozen=True, slots=True)
class RecordingResult:
    path: str
    """Final on-disk path of the produced MKV."""

    frames: int
    """Total frames muxed into the file."""

    bytes: int
    """On-disk size of the MKV in bytes."""

    duration_s: float
    """Wall-clock duration of the recording in seconds."""

    dropped: int
    """Frames the recorder dropped (writer back-pressure). 0 in
    healthy runs."""

    error: Optional[str] = None
    """Writer-crash description when the writer thread died before
    ``stop()`` — the file at :attr:`path` is likely truncated or
    incomplete. ``None`` for a clean recording."""


@dataclass(frozen=True, slots=True)
class PipelineMetrics:
    """Snapshot of pipeline performance — produced by `pipeline.metrics`."""
    camera_fps: float
    """EWMA-smoothed source-grab rate. Decays toward 0 if the source
    stalls (no tick for >2 seconds)."""

    inference_fps: float
    """EWMA-smoothed pipeline output rate (post-inference,
    post-postprocess)."""

    camera_drops: int
    """Cumulative frames the source pump dropped because the worker
    hadn't picked up the previous one. Reset on `seek()`."""

    output_drops: int
    """Cumulative frames the worker dropped pushing to the consumer-facing
    output queue (``pipeline.get()``) when it was full and policy is
    ``drop_oldest``. Indicates the user is reading slower than the
    pipeline produces — not an inference-engine issue. Reset on
    `seek()`."""

    camera_drops_per_s: float
    """EWMA-smoothed camera-drop rate (drops/second). Companion to
    :attr:`camera_drops` — total damage vs. current health. Decays
    toward 0 if no drops fire for >2 seconds."""

    output_drops_per_s: float
    """EWMA-smoothed output-drop rate (drops/second). Companion to
    :attr:`output_drops`. Useful as a live indicator of consumer
    keeping up; total counter is the post-mortem."""

    decode_p50_us: int
    """p50 latency of decode + SBS split + canvas remap (per-frame),
    microseconds."""

    rectify_p50_us: int
    """p50 latency of the rectify cv2.remap pass (subset of decode),
    microseconds."""

    inference_p50_us: int
    """p50 inference latency (engine submit → engine done),
    microseconds."""

    inference_p95_us: int
    """p95 inference latency — the long-tail indicator."""

    postprocess_p50_us: int
    """p50 postprocess latency (model.postprocess + filters + depth
    + cloud), microseconds."""

    worker_loop_p50_us: int
    """p50 worker-loop latency: from grab() return through decode,
    rectify, inference submit, and backpressure-wait — i.e. the time
    one iteration of the worker thread takes. Does NOT include source
    pump residence (frame waiting in driver queue) nor post-inference
    consumer latency. Microseconds."""

    worker_loop_p95_us: int
    """p95 worker-loop latency — long-tail indicator for whether the
    worker is keeping up."""

    inference_in_flight: int
    """Engine futures currently in flight. Bounded by `pipeline_depth`."""
