# API reference

Auto-generated from the package's public surface (the union of
`__all__` across `depz_camera_sdk` and its subpackages) by
`scripts/gen_api_md.py` — run `make docs` to regenerate. Edit the
docstrings in the source, not this file.

## Contents

- **Pipeline**: [`DepthPipeline`](#depthpipeline), [`FramePacket`](#framepacket), [`RecordingStatus`](#recordingstatus), [`RecordingResult`](#recordingresult), [`PipelineMetrics`](#pipelinemetrics)
- **Sources**: [`FrameSource`](#framesource), [`CameraSource`](#camerasource), [`MkvSource`](#mkvsource), [`CameraInfo`](#camerainfo), [`RawFrame`](#rawframe), [`resolve_source_arg`](#resolve_source_arg)
- **Inference**: [`DepthInferenceBase`](#depthinferencebase), [`NeuralStereoInferenceBase`](#neuralstereoinferencebase), [`NeuralMonoInferenceBase`](#neuralmonoinferencebase), [`SgbmInference`](#sgbminference), [`StereoBMInference`](#stereobminference), [`RectifyOnlyInference`](#rectifyonlyinference), [`build_inference`](#build_inference), [`available_inferences`](#available_inferences), [`register_inference`](#register_inference), [`DepzStereoInference`](#depzstereoinference), [`LightStereoInference`](#lightstereoinference), [`CREStereoInitialInference`](#crestereoinitialinference), [`CREStereoCombinedInference`](#crestereocombinedinference), [`CFastFoundationStereoInference`](#cfastfoundationstereoinference), [`DepthAnythingV2SmallInference`](#depthanythingv2smallinference), [`Metric3DV2VitSmallInference`](#metric3dv2vitsmallinference), [`InferenceVariant`](#inferencevariant), [`ModelFile`](#modelfile), [`get_inference`](#get_inference), [`cache_root`](#cache_root)
- **Engines**: [`available_engines`](#available_engines), [`list_engines`](#list_engines), [`recommend_engine`](#recommend_engine), [`InferenceEngine`](#inferenceengine), [`InferenceFuture`](#inferencefuture), [`EventFuture`](#eventfuture), [`ENGINE_CLASSES`](#engine_classes), [`class_rank`](#class_rank), [`classify_engine`](#classify_engine), [`make_engine`](#make_engine), [`parse_engine_slug`](#parse_engine_slug), [`priority_rank`](#priority_rank)
- **Calibration**: [`Calibration`](#calibration), [`CanvasLayout`](#canvaslayout), [`OpenCVBasicCalibration`](#opencvbasiccalibration), [`OpenCVFisheyeCalibration`](#opencvfisheyecalibration), [`register_calibration`](#register_calibration), [`registered_polymorphic_names`](#registered_polymorphic_names)
- **Filters**: [`Filter`](#filter), [`DisparityFilter`](#disparityfilter), [`DepthFilter`](#depthfilter), [`PointCloudFilter`](#pointcloudfilter), [`SpeckleFilter`](#specklefilter), [`ZClipFilter`](#zclipfilter), [`EdgeFilter`](#edgefilter), [`TemporalEmaFilter`](#temporalemafilter), [`BoundingBoxFilter`](#boundingboxfilter), [`StatisticalOutlierFilter`](#statisticaloutlierfilter), [`default_filter_chain`](#default_filter_chain)
- **Cameras**: [`fetch_camera_info_json`](#fetch_camera_info_json), [`DepzCameraInfo`](#depzcamerainfo), [`list_connected_depz_cameras`](#list_connected_depz_cameras), [`get_first_connected_depz_camera`](#get_first_connected_depz_camera), [`get_connected_depz_camera_by_serial`](#get_connected_depz_camera_by_serial), [`CaptureMode`](#capturemode), [`CameraControl`](#cameracontrol), [`CameraSpec`](#cameraspec), [`DEPZ_VARIUS`](#depz_varius), [`DEPZ_MINIMA`](#depz_minima), [`DEPZ_CLARITY`](#depz_clarity), [`REGISTRY`](#registry), [`spec_by_sku`](#spec_by_sku), [`list_connected_depz_cameras_detailed`](#list_connected_depz_cameras_detailed)
- **Platform**: [`PlatformBackend`](#platformbackend), [`CameraControls`](#cameracontrols), [`VideoDeviceInfo`](#videodeviceinfo), [`current_backend`](#current_backend)
- **Controls**: [`Control`](#control)
- **Cache**: [`get_cache_dir`](#get_cache_dir), [`set_cache_dir`](#set_cache_dir)
- **Logging_Setup**: [`configure_logging`](#configure_logging), [`enable_default_logging`](#enable_default_logging)

## Pipeline

### DepthPipeline

```python
class DepthPipeline(source: 'FrameSource | str | Path | Any | None' = None, *, inference: 'DepthInferenceBase | tuple[str, str] | None' = None, enable_raw: bool = False, enable_rectified: bool = False, enable_disparity: bool = False, enable_depth: bool = True, enable_cloud: bool = False, max_output_queue: int = 4, pipeline_depth: int = 2, output_drop_policy: str = 'drop_oldest', photo_mode: bool = False, recalibration_enabled: bool = False) -> None
```

Depth SDK pipeline. Single-shot lifecycle. Drives any
:class:`DepthInferenceBase` (stereo-disparity neural, classical SGBM,
monocular metric depth, rectify-only) over any
:class:`FrameSource` (live UVC camera or MKV replay).

Source-swap (live ↔ replay): destroy this instance, create a new one
with a fresh source. ``attach_source`` / ``detach_source`` NOT
supported.

Construction is lightweight (no model load, no source open). Heavy
work happens in :meth:`start`. Frames flow out via :meth:`get`.

Pipeline-level runtime controls follow the same descriptor pattern as
filters / inference / calibration — see :meth:`controls` for the
list. Mutations from RPC threads go through
:meth:`set_pipeline_control` (enqueues a worker command);
introspection via :meth:`pipeline_info`. Stream-graph gates
(``raw`` / ``rectified`` / ``disparity`` / ``depth`` / ``cloud``)
have a separate API (:meth:`set_stream_enabled`) because their
cascade semantics don't fit a flat coerce-and-clamp descriptor.

**Runtime controls** — read via the attribute, set via the owner's `set_*` API:

| control | type | default | range / values | unit |
|---|---|---|---|---|
| `photo_mode` | bool | `False` | — | — |
| `output_drop_policy` | str | `drop_oldest` | `drop_oldest`, `block` | — |
| `replay_calibration_mode` | str | `all` | `all`, `exclude_controls`, `none` | — |

#### DepthPipeline.is_running *(property)*

#### DepthPipeline.last_error *(property)*

Crash detail of the worker thread — ``{"message", "traceback"}``
— or ``None`` while healthy. Set when the worker dies on an
unhandled exception; cleared by :meth:`start`. Poll together with
:attr:`is_running` to distinguish "stopped on request" from
"crashed" (UI surfaces the latter as a pipeline_error toast).

#### DepthPipeline.camera_info *(property)*

#### DepthPipeline.list_filters

```python
list_filters(self) -> list[dict]
```

Self-describing snapshot of every filter in the chain.

Each entry: ``{"slug", "label", "operates_on", "controls",
"state"}``. ``controls`` is the static descriptor list (drives
UI rendering); ``state`` is the live control values right now.
Use :meth:`set_filter_control` to mutate.

#### DepthPipeline.inference_info

```python
inference_info(self) -> dict
```

Self-describing snapshot of the active DepthInferenceBase.

Returns ``{"slug", "label", "input_size", "controls", "state"}``.
``controls`` is the static descriptor list (drives UI rendering);
``state`` is the live control values. Use
:meth:`set_inference_control` to mutate.

#### DepthPipeline.recalibration_info

```python
recalibration_info(self) -> Optional[dict]
```

Self-describing snapshot of the online-recalibration controller, or
``None`` when the pipeline is mono / recalibration is unsupported.

Mirrors :meth:`inference_info` / :meth:`list_filters` shape:
``{"slug", "controls", "state"}``. ``controls`` is the static
descriptor list (drives UI rendering — the tuning knobs + the
``enabled`` on/off); ``state`` is the live control values. Use
:meth:`set_recalibration_control` to mutate. The *runtime* state
(state machine, health, last_event) is the separate
:attr:`recalibration_status` surface — config here, observability
there, the same split as ``pipeline_info`` vs ``metrics``.

#### DepthPipeline.control_set

```python
control_set(self, name: str, value: Any) -> Any
```

Worker-side control apply (ControlOwner contract). The base impl
validates + stores via ``setattr``; this override only adds
the per-control side effect (clearing ``_snapshot_pending`` when
leaving photo mode).

DO NOT call from RPC threads — the pipeline's mutable state is
worker-owned, and direct mutation races with the running
worker. Use :meth:`set_pipeline_control` from outside the
worker; it routes through the command queue and waits.

#### DepthPipeline.pipeline_info

```python
pipeline_info(self) -> dict
```

Self-describing snapshot of pipeline-level controls. Mirrors
:meth:`inference_info` / :meth:`list_filters` shape: ``controls``
is the static descriptor list, ``state`` is the live values.

#### DepthPipeline.set_pipeline_control

```python
set_pipeline_control(self, name: str, value: Any) -> Any
```

Mutate one pipeline-level control from an RPC thread. See
:meth:`controls` for the current set. Enqueues a worker
command and waits.

Validation runs synchronously on the caller's thread (so
``KeyError`` / ``ValueError`` raise before enqueuing). Returns
the value actually applied. Same wait-for-worker pattern as
:meth:`set_filter_control` / :meth:`set_inference_control`.

#### DepthPipeline.active_calibration *(property)*

The calibration in effect right now —
:attr:`calibration_override` if set, otherwise
:attr:`base_calibration`. This is what the worker reads on
each frame and what consumers should use for any active
geometry / projection question.

#### DepthPipeline.canvas_fit_mode *(property)*

#### DepthPipeline.streams_enabled *(property)*

#### DepthPipeline.metrics *(property)*

#### DepthPipeline.recording_status *(property)*

#### DepthPipeline.recalibration_status *(property)*

Snapshot of the background online-recalibration loop, or None when
the pipeline is mono / recalibration is unsupported. Read-only; the
returned dataclass is reachable via attribute access (no import needed
— concealed, not exported in ``__init__``).

#### DepthPipeline.recalibration_history

```python
recalibration_history(self, since: float = 0.0)
```

#### DepthPipeline.recalibration_buffer_meta

```python
recalibration_buffer_meta(self)
```

#### DepthPipeline.recalibration_buffer_points

```python
recalibration_buffer_points(self, slot: int)
```

#### DepthPipeline.recalibration_buffer_jpeg

```python
recalibration_buffer_jpeg(self, slot: int, eye: str = 'l', max_w: int = 0)
```

#### DepthPipeline.recalibration_clear_service

```python
recalibration_clear_service(self)
```

Manual override: clear the sticky service_advised flag + failure
count (the panel's "needs service" label click).

#### DepthPipeline.start

```python
start(self) -> None
```

Open source, prepare engine, spawn worker.

#### DepthPipeline.stop

```python
stop(self) -> None
```

#### DepthPipeline.reset_camera_source

```python
reset_camera_source(self) -> bool
```

Close and reopen the underlying capture device in place.

Recovers a wedged camera — e.g. the UVC auto-exposure fault that
emits uniform frames until the device is power-cycled (the
"unplugging fixes it" bug). Live sources only; a no-op (returns
``False``) for replay/mock sources. The worker keeps running: its
``grab()`` returns ``None`` during the brief reopen gap, which it
already tolerates, so this is safe to call on a running pipeline.

Frame geometry is unchanged across a same-device reopen, so no
calibration rebuild is needed. Returns ``True`` if a reopen was
performed.

#### DepthPipeline.get

```python
get(self, timeout: Optional[float] = None) -> Optional[depz_camera_sdk.pipeline.types.FramePacket]
```

Pop the next :class:`FramePacket` from the output queue.

Default ``timeout=None`` blocks until a packet arrives (or the
pipeline shuts down). Pass an explicit ``timeout=<seconds>`` to
treat a slow / hung worker as a watchdog event — the call
returns ``None`` when no packet shows up in that window.

Note: pipeline-level *controls* are read via
:meth:`control_get(name)` (the :class:`~depz_camera_sdk._controls.ControlOwner`
contract). Different namespace, different prefix — no
collision.

#### DepthPipeline.get_snapshot

```python
get_snapshot(self, timeout: Optional[float] = None) -> Optional[depz_camera_sdk.pipeline.types.FramePacket]
```

Wait for the result of a photo-mode snapshot, dropping any
live-preview bypass packets in the meantime. Returns the first
:class:`FramePacket` whose :attr:`FramePacket.is_snapshot`
is True, or None on timeout (or pipeline stop).

Typical usage::

    pipeline.set_pipeline_control("photo_mode", True)
    pipeline.start()
    ...
    pipeline.trigger_snapshot()
    result = pipeline.get_snapshot(timeout=30.0)

Outside photo mode no packet ever has ``is_snapshot=True`` —
:meth:`get_snapshot` will block until the timeout expires. If
you want "next packet, snapshot or not," use :meth:`get`.

``timeout`` is the OVERALL deadline in seconds (None = wait
forever). Bypass packets dropped en route don't reset it.

#### DepthPipeline.set_filter_control

```python
set_filter_control(self, filter_slug: str, name: str, value: Any) -> Any
```

Mutate one filter control, e.g.
``set_filter_control("speckle", "max_size", 80)``. Goes through the
worker thread's command queue. Returns the value actually
stored (after coercion + clamp by the filter).

#### DepthPipeline.set_inference_control

```python
set_inference_control(self, name: str, value: Any) -> Any
```

Mutate one DepthInferenceBase control, e.g.
``set_inference_control("num_disparities", 128)`` for SGBM.
Routed through the worker so backends rebuild their internal
state (cv2.StereoSGBM matcher) on the worker thread, never on
the API thread. Returns the value actually stored (after
coercion + clamp by the inference).

#### DepthPipeline.set_recalibration_control

```python
set_recalibration_control(self, name: str, value: Any) -> Any
```

Mutate one online-recalibration control, e.g.
``set_recalibration_control("enabled", False)`` to pause the loop or
``set_recalibration_control("ok_px", 0.8)`` to retune the trigger.
Routed through the worker so the controller's ``control_set`` side
effects (rebuilding the health monitor on a ``window``/``ok_px``
change) run off the API thread. Returns the value actually stored
(after coercion + clamp by the controller). Mirrors
:meth:`set_inference_control`. Raises ``RuntimeError`` when the
pipeline is mono / recalibration is unsupported (the controller is
``None``) — introspect via :meth:`recalibration_info` first.

#### DepthPipeline.set_calibration_control

```python
set_calibration_control(self, name: str, value: Any) -> Any
```

Generic calibration-control mutation. Concrete `Calibration` impl
decides which control names are valid (e.g. opencv_basic accepts
'rectify_alpha'). Drains in-flight inference + rebuilds canvas
remap + resets stateful filters. Returns the value actually
applied (after coercion + clamp by the calibration impl).

#### DepthPipeline.update_calibration

```python
update_calibration(self, patch: dict) -> None
```

Apply an online-recalibrator patch. `patch` is opaque to the
SDK — `Calibration.apply_update` interprets it. Same drain +
rebuild + filter-reset path as `set_calibration_control`.

#### DepthPipeline.trigger_snapshot

```python
trigger_snapshot(self) -> None
```

Arm a one-shot inference for the next frame. Only meaningful
when ``photo_mode`` is enabled (set via
:meth:`set_pipeline_control`) — outside photo mode inference
runs every frame, so this is a no-op.

Idempotent: repeated calls before the snapshot fires don't
queue multiple inferences. Each fired snapshot disarms the
flag, so re-clicking Snap after the result lands re-arms it.

Fire-and-forget — does NOT wait for the worker to process the
command. That's deliberate: if the user spams Snap during a
slow inference, all triggers should land in the command queue
as a batch so the handler's "already pending" dedup
coalesces them. A blocking wait would serialize the spam
one-frame-apart and defeat the dedup.

#### DepthPipeline.set_stream_enabled

```python
set_stream_enabled(self, name: str, enabled: bool, *, cascade: bool = False) -> tuple[bool, str]
```

Toggle one stream on/off in the FramePacket emission graph.

Enable always cascades upstream dependencies (you can't compute
depth without disparity, so enabling depth flips disparity on
too). Disable defaults to refusing if a downstream still
consumes the stream — explicit `cascade=True` flips them all
off in one shot. The set of cascaded-off stream names is
returned to the caller via the command (and exposed here in
the second tuple element, replacing `reason` when applied).

Safe to call **before** `start()` — there's no worker thread
yet, so we mutate the stream-graph dict directly. After
`start()` the call routes through the command queue so the
worker applies it between frames.

#### DepthPipeline.step_replay

```python
step_replay(self, direction: str = 'forward') -> None
```

Advance / rewind exactly one frame in MKV replay.

Unlike :meth:`seek`, this doesn't compute a PTS target and
doesn't reset stateful filters. The source's libav demuxer
moves by exactly one packet (one decoded frame), avoiding
keyframe-rounding / fps-jitter artifacts that PTS-target
seeks suffer from.

Caller should pause the replay first (otherwise normal
playback continues to overwrite the stepped frame). Raises
if `source.is_seekable=False`.

#### DepthPipeline.seek

```python
seek(self, *, monotonic_ns: Optional[int] = None, utc_ns: Optional[int] = None, relative_us: Optional[int] = None, fraction: Optional[float] = None) -> None
```

#### DepthPipeline.is_pausable *(property)*

#### DepthPipeline.is_paused *(property)*

#### DepthPipeline.pause

```python
pause(self) -> None
```

Pause the source. Raises if the source is not pausable
(live cameras, mocks).

Metric freezing follows automatically: the worker thread
syncs ``metrics.frozen`` to ``source.is_paused`` on every
iter — so this also covers MkvSource auto-pausing on EOF,
not just explicit RPC pauses. See
:meth:`_worker_loop_inner` for the reset-on-play-edge logic
that keeps every play session starting at zero.

#### DepthPipeline.resume

```python
resume(self) -> None
```

Resume a paused source. Raises if not pausable. Worker
zeroes metrics on the paused→playing edge — see
:meth:`pause` for the rationale.

#### DepthPipeline.t0_monotonic_ns *(property)*

CLOCK_MONOTONIC anchor of the active source (the wall time
of the source's first emitted frame). `pts_us=0` corresponds
to this anchor. None on a live source until its pump has
produced the first frame.

#### DepthPipeline.t0_utc_ns *(property)*

UTC anchor of the active source. None for live sources
without a wall-clock anchor; replay sources read this from
the recording's metadata.

#### DepthPipeline.pts_to_monotonic_ns

```python
pts_to_monotonic_ns(self, pts_us: int) -> Optional[int]
```

Convert a packet's pts_us to absolute CLOCK_MONOTONIC
nanoseconds. Returns None when no anchor is available.

#### DepthPipeline.pts_to_utc_ns

```python
pts_to_utc_ns(self, pts_us: int) -> Optional[int]
```

Convert a packet's pts_us to absolute UTC nanoseconds.
Returns None when no UTC anchor is available.

#### DepthPipeline.set_camera_control

```python
set_camera_control(self, name: str, value: float) -> bool
```

Direct write to the camera-control fd. Bypasses the cmd-queue
because V4L2 ioctl is independent of pipeline state — it
touches a separate fd held by the source. Returns ``False`` for
replay sources (no live fd to write to).

#### DepthPipeline.get_camera_control

```python
get_camera_control(self, name: str) -> Optional[float]
```

Read one named camera-control value.

Live source: V4L2 ioctl read off the source's parallel fd.
Replay source: fall back to the history-driven snapshot via
:meth:`FrameSource.list_camera_controls` so consumers see the
same shape and the value that was active at the most recently
emitted frame's PTS.

#### DepthPipeline.list_camera_controls

```python
list_camera_controls(self)
```

Snapshot of every camera control available on the
active source (live cameras only — replay sources return ``[]``).

Returns ``list[CameraControl]`` — public, platform-neutral
descriptors with live ``current_value`` / ``min`` / ``max`` /
``step``. Internal V4L2 / Win32 IDs are stripped. Use
:meth:`set_camera_control` / :meth:`get_camera_control` to
write/read individual controls by name afterwards.

Re-call to refresh — values are point-in-time snapshots.

#### DepthPipeline.start_recording

```python
start_recording(self, path: str | pathlib.Path, camera_controls: Optional[dict] = None) -> None
```

Start recording an MKV at ``path``.

The camera-info payload (the JSON the device-info endpoint
returned, embedded verbatim as an MKV attachment) is sourced
automatically from whatever the source is bound to:

- Live :class:`CameraSource` constructed with ``camera_info=…``
  → uses ``source.camera_info.to_dict()``.
- :class:`MkvSource` → uses its embedded ``camera_info.json``.
- Anything else (custom source, no DepzCameraInfo) →
  synthesizes from ``self.active_calibration.to_dict()`` + identity.

``camera_controls`` is an optional caller override for the
initial camera-state snapshot; default comes from the
controller's provider callback.

#### DepthPipeline.stop_recording

```python
stop_recording(self) -> depz_camera_sdk.pipeline.types.RecordingResult
```

#### DepthPipeline.pause_recording

```python
pause_recording(self) -> None
```

Pause the active recording — stop writing frames while keeping
the output file open. Resumes the same recording on
:meth:`resume_recording`. No-op when not recording.

Note: the recorder is fed internally by the worker, so pause/resume
can only be driven from inside the SDK — it can't be gated by an
external consumer. Kept here for that reason.

#### DepthPipeline.resume_recording

```python
resume_recording(self) -> None
```

Resume a paused recording into the same file.

#### DepthPipeline.control_get

```python
control_get(self, name: str) -> Any
```

Return the current value of one control by name — reads its
``_control_<name>`` backing attribute. Equivalent to ``host.<name>``
(the descriptor) but takes the name as a string.

#### DepthPipeline.controls_to_dict

```python
controls_to_dict(self) -> dict
```

Snapshot every current control value — for the UI live-state
pane, recorder sidecar, etc.

#### DepthPipeline.controls

```python
controls(self) -> tuple[depz_camera_sdk._controls.Control, ...]
```

This host's control descriptors — the class-level
:data:`_CONTROL_SPECS` tuple (same object every call, no per-call
allocation). The single public accessor shared by every
ControlOwner: filters, inference, calibration, pipeline.

### FramePacket

```python
class FramePacket(frame_number: int, pts_us: int, source_frame_idx: Optional[int] = None, left: Optional[numpy.ndarray] = None, right: Optional[numpy.ndarray] = None, rectified_left: Optional[numpy.ndarray] = None, rectified_right: Optional[numpy.ndarray] = None, disparity: Optional[numpy.ndarray] = None, depth: Optional[numpy.ndarray] = None, point_cloud: Optional[numpy.ndarray] = None, is_snapshot: bool = False) -> None
```

One processed stereo frame — what `pipeline.get()` returns.

All array fields are optional and may be None when their stream has
been disabled via `set_stream_enabled`. Shapes/dtypes are documented
on each field.

### RecordingStatus

```python
class RecordingStatus(is_recording: bool, path: Optional[str], frames_written: int, bytes_written: int, dropped: int, duration_s: float, is_paused: bool = False, error: Optional[str] = None) -> None
```

### RecordingResult

```python
class RecordingResult(path: str, frames: int, bytes: int, duration_s: float, dropped: int, error: Optional[str] = None) -> None
```

### PipelineMetrics

```python
class PipelineMetrics(camera_fps: float, inference_fps: float, camera_drops: int, output_drops: int, camera_drops_per_s: float, output_drops_per_s: float, decode_p50_us: int, rectify_p50_us: int, inference_p50_us: int, inference_p95_us: int, postprocess_p50_us: int, worker_loop_p50_us: int, worker_loop_p95_us: int, inference_in_flight: int) -> None
```

Snapshot of pipeline performance — produced by `pipeline.metrics`.

## Sources

### FrameSource

```python
class FrameSource()
```

Frame source — yields raw frames (SBS-stereo or single-eye-mono)
one at a time. Two concrete impls: :class:`CameraSource` (live UVC),
:class:`MkvSource` (recording replay).

Lifecycle: ``open()`` once → ``grab()`` until None (EOF) →
``close()``. Sources are NOT thread-safe; the pipeline owns and
uses them from the worker thread only (CameraSource's internal
pump thread is an implementation detail; consumers see the
single-threaded surface).

#### FrameSource.open

```python
open(self) -> depz_camera_sdk.sources.types.CameraInfo
```

Open underlying device or file. Returns geometry/identity.
Raises on failure.

#### FrameSource.grab

```python
grab(self, timeout: float = 1.0) -> Optional[depz_camera_sdk.sources.types.RawFrame]
```

Return next packet. None signals EOF (only for bounded sources
like MkvSource — CameraSource never returns None unless an error
occurs, in which case it raises).

#### FrameSource.close

```python
close(self) -> None
```

Release resources. Idempotent.

#### FrameSource.is_live *(property)*

#### FrameSource.is_seekable *(property)*

#### FrameSource.is_mono *(property)*

True when the source delivers a single-eye frame (no SBS
split). Live cameras read this from the active
:class:`CaptureMode`; replay sources read it from the embedded
calibration's :attr:`Calibration.is_stereo` (inverted). Default
False keeps existing stereo sources unaffected.

#### FrameSource.fourcc *(property)*

Wire-format fourcc of `RawFrame.jpeg` payload. Default 'MJPG'.
Sources producing other byte layouts (YUYV, NV12, ...) override.

#### FrameSource.decode_to_bgr

```python
decode_to_bgr(self, raw: depz_camera_sdk.sources.types.RawFrame) -> Optional[numpy.ndarray]
```

Decode the wire bytes into a BGR HxWx3 uint8 array.

Default implementation handles MJPG (cv2.imdecode reads dims
from the JPEG header) and YUYV (raw layout — frame size taken
from the source's :attr:`_info`, populated in ``open()``).
Sources with non-standard byte layouts override.

Returns None on decode failure (caller skips the frame).

#### FrameSource.camera_controls *(property)*

Camera-control handle (CameraControls). None for replay/mock.

#### FrameSource.list_camera_controls

```python
list_camera_controls(self) -> list[depz_camera_sdk.cameras.camera_spec.CameraControl]
```

Snapshot of the camera controls available on this
source. Returns ``[]`` for sources that don't expose controls
(replay, mocks). Live cameras override.

Each :class:`CameraControl` carries a *snapshot* of `current_value`
/ `min` / `max` / `step` at the moment of the call — these may
change over time (auto-exposure, manual writes). Re-call to
refresh.

Controls not actually available on the current platform are
omitted. The returned list is therefore "what you can really
use right now" — safe to drive a UI from directly.

#### FrameSource.mode *(property)*

CaptureMode the source was configured with. None when irrelevant.

#### FrameSource.add_fanout

```python
add_fanout(self, cb: Callable[[depz_camera_sdk.sources.types.RawFrame], NoneType]) -> None
```

Register a callback invoked on every grabbed frame at full
source rate (BEFORE drop-to-latest). Used by recorder. Default:
not supported — recorder will fall back to worker-rate feeding.

#### FrameSource.remove_fanout

```python
remove_fanout(self, cb: Callable[[depz_camera_sdk.sources.types.RawFrame], NoneType]) -> None
```

Inverse of add_fanout. Default: no-op.

#### FrameSource.get_pump_stats

```python
get_pump_stats(self) -> tuple[int, int]
```

Return (frames_grabbed, frames_dropped) since last call.
For sources without an internal pump thread (replay, mocks),
returns (1, 0) per call so the pipeline metrics still tick.

Worker calls this once per consumed frame; counter resets on read.

#### FrameSource.base_calibration *(property)*

The calibration as loaded at :meth:`open` time — frozen for
the lifetime of this source. ``None`` if the source doesn't
ship one.

For :class:`CameraSource`: the calibration bound to the
:class:`DepzCameraInfo` (server-shape device-info payload).
For :class:`MkvSource`: parsed once from the recording's
``camera_info.json`` attachment.

Pair with :attr:`active_calibration` for the in-effect-now one
(which differs only on replay sources that crossed an
online-recal history boundary).

#### FrameSource.active_calibration *(property)*

The calibration in effect at the source's *current*
playback position, or ``None`` if the source doesn't ship one.

For replay sources whose calibration was auto-tuned during
recording, this advances as the source crosses online-recal
history-entry boundaries (in :meth:`grab` /
:meth:`seek_relative_us`), so consumers always see what was
active at the most recently emitted frame's PTS. Without
history, equals :attr:`base_calibration`. For live cameras
also equals :attr:`base_calibration` (no playback timeline).

Stable identity: when the history epoch hasn't changed,
consecutive reads return the *same instance* — consumers can
cheap-detect "nothing changed" with an ``is`` compare instead
of a dict diff.

#### FrameSource.duration_us *(property)*

Total duration in microseconds, or None for unbounded (live) sources.

#### FrameSource.camera_controls_snapshot *(property)*

For MkvSource: camera-props snapshot at recording start.

#### FrameSource.mode_slug *(property)*

Round-trip slug for the capture mode this source emits (see
:attr:`CaptureMode.slug`). Live cameras supply their bound
mode's slug; replay sources read it back from the
``CAPTURE_MODE`` Matroska tag. ``None`` when the source
doesn't model a discrete mode (mocks).

#### FrameSource.t0_monotonic_ns *(property)*

Monotonic-clock anchor for ``RawFrame.pts_us``. Live cameras
set this on the first emitted frame; replay sources set it
eagerly from the recording's first packet timestamp. ``None``
until the source has produced its first frame.

#### FrameSource.t0_utc_ns *(property)*

UTC anchor for ``RawFrame.pts_us`` in nanoseconds. Replay
sources read this from the recording's metadata; live cameras
without a wall-clock anchor return ``None``.

#### FrameSource.pts_to_monotonic_ns

```python
pts_to_monotonic_ns(self, pts_us: int) -> Optional[int]
```

Convert a stream-relative `pts_us` to absolute CLOCK_MONOTONIC
nanoseconds on the source's clock domain. Returns None when the
source has no monotonic anchor (e.g. mock source pre-open).

Default impl works for any source that publishes
`t0_monotonic_ns`; concrete sources can override if they need
a non-linear mapping (none currently do).

#### FrameSource.pts_to_utc_ns

```python
pts_to_utc_ns(self, pts_us: int) -> Optional[int]
```

Convert pts_us to absolute UTC nanoseconds. Default impl uses
`t0_utc_ns`; returns None if no UTC anchor.

#### FrameSource.seek_relative_us

```python
seek_relative_us(self, pts_us: int) -> None
```

Seek to a stream-relative microsecond offset. Default raises.

#### FrameSource.is_pausable *(property)*

True iff `pause()` / `resume()` are meaningful for this source.
Live cameras hardware-pace and can't be paused without dropping
frames; only replay sources implement pause.

#### FrameSource.is_paused *(property)*

Current pause state. Always False for non-pausable sources.

#### FrameSource.pause

```python
pause(self) -> None
```

Pause the source — `grab()` blocks until `resume()`. Default
raises so live-camera consumers fail fast.

#### FrameSource.resume

```python
resume(self) -> None
```

Inverse of `pause()`. Default raises.

### CameraSource

```python
class CameraSource(camera_info: 'DepzCameraInfo', mode: 'CaptureMode | None' = None, initial_controls: 'dict[str, float] | None' = None, output_eye: 'tuple[int, int] | None' = None) -> None
```

Live camera capture. Constructor parameters describe the desired
capture mode; `open()` actually grabs the device.

#### CameraSource.os_id *(property)*

#### CameraSource.mode *(property)*

#### CameraSource.mode_slug *(property)*

Round-trip identifier for the capture mode (see
:attr:`CaptureMode.slug`). Live source returns its bound mode's
slug; recorder embeds it as a Matroska tag so replay can pick
the same :class:`CaptureMode` from the matched
:class:`CameraSpec`.

#### CameraSource.camera_controls *(property)*

#### CameraSource.camera_info *(property)*

The resolved camera identity bound to this source. Always
non-None: :meth:`__init__` rejects raw paths so this is
guaranteed to carry path + sku + spec + calibration. Read by
:meth:`DepthPipeline.start_recording` to embed the camera-info
payload as an MKV attachment without the caller having to
plumb anything separately.

#### CameraSource.base_calibration *(property)*

Bound calibration — same object as
:attr:`camera_info.base_calibration`. Live cameras have no
playback timeline so :attr:`active_calibration` returns the
same instance.

#### CameraSource.active_calibration *(property)*

Same as :attr:`base_calibration` for live sources — no
replay timeline, no history catch-up.

#### CameraSource.list_camera_controls

```python
list_camera_controls(self) -> list[depz_camera_sdk.cameras.camera_spec.CameraControl]
```

Snapshot of every camera control on this device.

The matched :class:`CameraSpec` is the single source of truth for
what controls exist and their valid ranges (``min`` / ``max`` /
``step`` / ``default`` / ``label`` / ``unit`` / ``enum_values``
/ ``kind`` / cascade gates). The platform handle is used only to
snapshot the *current* value — driver QUERYCTRL fallbacks are
gone, the spec is authoritative.

Returns ``[]`` when the source isn't open or no CameraSpec is
matched.

#### CameraSource.is_live *(property)*

#### CameraSource.is_seekable *(property)*

#### CameraSource.t0_monotonic_ns *(property)*

CLOCK_MONOTONIC anchor of the first emitted frame, or None
until the pump has produced its first packet. Once set, stays
constant for the lifetime of the source. `pts_us=0` corresponds
to this anchor.

#### CameraSource.fourcc *(property)*

#### CameraSource.open

```python
open(self) -> depz_camera_sdk.sources.types.CameraInfo
```

#### CameraSource.add_fanout

```python
add_fanout(self, cb: Callable[[depz_camera_sdk.sources.types.RawFrame], NoneType]) -> None
```

Register callback called on every grabbed frame at full camera rate.
Callback runs on the pump thread — must be cheap (typically
queue.put_nowait). Used by recorder so dataset gets full-rate
even when inference can't keep up.

#### CameraSource.remove_fanout

```python
remove_fanout(self, cb: Callable[[depz_camera_sdk.sources.types.RawFrame], NoneType]) -> None
```

#### CameraSource.decode_to_bgr

```python
decode_to_bgr(self, raw: depz_camera_sdk.sources.types.RawFrame) -> 'Optional[np.ndarray]'
```

Decode wire bytes to BGR, then resize to the delivered size
when an ``output_eye`` target is set. Overrides the base impl
because YUYV reshape must use the CAPTURED dims (``_info`` may
already report the scaled size).

#### CameraSource.grab

```python
grab(self, timeout: float = 1.0) -> Optional[depz_camera_sdk.sources.types.RawFrame]
```

Pop the latest available frame (drops older ones).

Blocks up to `timeout` seconds for a frame to appear if buffer
empty. Returns None on timeout (camera silent for `timeout`
seconds — likely device error).

During a :meth:`DepthPipeline.reset_camera_source` reopen gap the
source is transiently closed — grab() returns ``None`` then (the
worker already tolerates None), it does NOT raise. Raising here
killed the worker whenever the startup wedge-recovery reset the
device mid-stream. A never-opened source still raises — that's a
caller bug, not a transient state.

#### CameraSource.get_pump_stats

```python
get_pump_stats(self) -> tuple[int, int]
```

Return (frames_grabbed, frames_dropped) since last call.
Both counters reset on each call. Worker polls this to feed the
pipeline's MetricsCollector — keeps camera_fps separate from
inference_fps.

#### CameraSource.close

```python
close(self) -> None
```

#### CameraSource.is_mono *(property)*

True when the source delivers a single-eye frame (no SBS
split). Live cameras read this from the active
:class:`CaptureMode`; replay sources read it from the embedded
calibration's :attr:`Calibration.is_stereo` (inverted). Default
False keeps existing stereo sources unaffected.

#### CameraSource.duration_us *(property)*

Total duration in microseconds, or None for unbounded (live) sources.

#### CameraSource.camera_controls_snapshot *(property)*

For MkvSource: camera-props snapshot at recording start.

#### CameraSource.t0_utc_ns *(property)*

UTC anchor for ``RawFrame.pts_us`` in nanoseconds. Replay
sources read this from the recording's metadata; live cameras
without a wall-clock anchor return ``None``.

#### CameraSource.pts_to_monotonic_ns

```python
pts_to_monotonic_ns(self, pts_us: int) -> Optional[int]
```

Convert a stream-relative `pts_us` to absolute CLOCK_MONOTONIC
nanoseconds on the source's clock domain. Returns None when the
source has no monotonic anchor (e.g. mock source pre-open).

Default impl works for any source that publishes
`t0_monotonic_ns`; concrete sources can override if they need
a non-linear mapping (none currently do).

#### CameraSource.pts_to_utc_ns

```python
pts_to_utc_ns(self, pts_us: int) -> Optional[int]
```

Convert pts_us to absolute UTC nanoseconds. Default impl uses
`t0_utc_ns`; returns None if no UTC anchor.

#### CameraSource.seek_relative_us

```python
seek_relative_us(self, pts_us: int) -> None
```

Seek to a stream-relative microsecond offset. Default raises.

#### CameraSource.is_pausable *(property)*

True iff `pause()` / `resume()` are meaningful for this source.
Live cameras hardware-pace and can't be paused without dropping
frames; only replay sources implement pause.

#### CameraSource.is_paused *(property)*

Current pause state. Always False for non-pausable sources.

#### CameraSource.pause

```python
pause(self) -> None
```

Pause the source — `grab()` blocks until `resume()`. Default
raises so live-camera consumers fail fast.

#### CameraSource.resume

```python
resume(self) -> None
```

Inverse of `pause()`. Default raises.

### MkvSource

```python
class MkvSource(path: str | pathlib.Path, replay_mode: str = 'realtime', loop: bool = False, build_frame_index: bool = True, pace_playback: bool = True) -> None
```

Replay an SDK-recorded MKV as a `FrameSource`.

On ``open()`` it reads the four JSON attachments embedded in the MKV
(``camera_info``, ``pipeline_config``, ``calibration_history``,
``camera_state_history``), then paces frames per ``replay_mode``
(``"realtime"`` = wall-clock pump thread + drop-to-latest;
``"every_frame"`` = synchronous, blocks on the consumer). Exposes
pause / resume, PTS seek, and exact single-step (when
``build_frame_index=True``). ``active_calibration`` advances during
playback as recorded online-recal patches apply.

#### MkvSource.is_live *(property)*

#### MkvSource.is_seekable *(property)*

#### MkvSource.fourcc *(property)*

Wire format of the recorded video stream. Cached from codec
metadata at open(); ``"MJPG"`` before open() (matches the ABC
default).

#### MkvSource.mode_slug *(property)*

Capture-mode slug recorded by the producer (read from the
``CAPTURE_MODE`` Matroska tag). ``None`` for older
recordings that don't carry the tag.

#### MkvSource.base_calibration *(property)*

Calibration parsed once at :meth:`open` from the
recording's ``camera_info.json`` attachment — frozen for the
lifetime of this source.

#### MkvSource.active_calibration *(property)*

Calibration active at the most recently emitted frame's PTS.

Walks the ``calibration`` subtitle track for the latest
full-state entry with ``rel_us <= pts_us``. Each entry is a
complete cereal-shape calibration (not a delta) — random
access is O(N) walk + O(1) build, no replay-from-base needed.
Falls back to :attr:`base_calibration` when no entries precede
the current PTS.

Maintained inside :meth:`grab` / :meth:`seek_relative_us` /
the step methods — consumers read it after those calls.

#### MkvSource.duration_us *(property)*

#### MkvSource.fps_target *(property)*

Rate requested when the file was recorded (FPS_TARGET tag).
Aspirational — playback does not use it.

#### MkvSource.fps_measured *(property)*

Rate the file actually plays at, derived from the frame-pts
index. ``None`` when the index was not built.

#### MkvSource.t0_monotonic_ns *(property)*

#### MkvSource.t0_utc_ns *(property)*

#### MkvSource.clock_domain *(property)*

#### MkvSource.camera_controls_snapshot *(property)*

#### MkvSource.list_camera_controls

```python
list_camera_controls(self) -> list
```

Replay-time camera-control snapshot — exactly the same shape
:class:`CameraSource.list_camera_controls` returns for live
cameras, so consumer UIs (web-viewer's camera panel) get live-
updating values during playback without special-casing replay.

Built from the matched :class:`CameraSpec`'s control literals
plus ``_current_camera_state`` (advanced per frame). Empty list
when no SKU / spec match (recordings from cameras the SDK
doesn't know).

#### MkvSource.open

```python
open(self) -> depz_camera_sdk.sources.types.CameraInfo
```

#### MkvSource.is_pausable *(property)*

#### MkvSource.pause

```python
pause(self) -> None
```

#### MkvSource.resume

```python
resume(self) -> None
```

#### MkvSource.is_paused *(property)*

#### MkvSource.grab

```python
grab(self, timeout: float = 1.0) -> Optional[depz_camera_sdk.sources.types.RawFrame]
```

#### MkvSource.add_fanout

```python
add_fanout(self, cb: Callable[[depz_camera_sdk.sources.types.RawFrame], NoneType]) -> None
```

Register callback called on every demuxed frame at full file
rate, BEFORE drop-to-latest. Active only in ``realtime`` mode —
in ``every_frame`` mode the pump never starts so callbacks are
never invoked (recorder falls back to worker-rate feeding).
Callback runs on the pump thread; must not block.

#### MkvSource.remove_fanout

```python
remove_fanout(self, cb: Callable[[depz_camera_sdk.sources.types.RawFrame], NoneType]) -> None
```

#### MkvSource.get_pump_stats

```python
get_pump_stats(self) -> tuple[int, int]
```

(grabbed, dropped) since last call. Realtime mode draws on
the pump's counters. every_frame mode has no pump → defer to
the ABC default (1, 0) so pipeline metrics still tick.

#### MkvSource.seek_relative_us

```python
seek_relative_us(self, pts_us: int) -> None
```

#### MkvSource.step_one_forward

```python
step_one_forward(self) -> None
```

Advance to frame at ``_current_idx + 1``.

Looks up the next frame's pts in the eager frame index built
at open(), seeks libav to it. Exact: every packet in MJPG /
rawvideo is a keyframe, so libav's "seek to keyframe ≤ target"
lands precisely on that frame. No fps-jitter or pts arithmetic.

Caller must pause the source first; the bypass flag emits one
packet through the paused gate. No-op at end of file.

Raises if the source was opened with ``build_frame_index=False``.

#### MkvSource.step_one_backward

```python
step_one_backward(self) -> None
```

Move to frame at ``_current_idx - 1``. No-op at frame 0.

Raises if the source was opened with ``build_frame_index=False``.

#### MkvSource.close

```python
close(self) -> None
```

#### MkvSource.is_mono *(property)*

True when the source delivers a single-eye frame (no SBS
split). Live cameras read this from the active
:class:`CaptureMode`; replay sources read it from the embedded
calibration's :attr:`Calibration.is_stereo` (inverted). Default
False keeps existing stereo sources unaffected.

#### MkvSource.decode_to_bgr

```python
decode_to_bgr(self, raw: depz_camera_sdk.sources.types.RawFrame) -> Optional[numpy.ndarray]
```

Decode the wire bytes into a BGR HxWx3 uint8 array.

Default implementation handles MJPG (cv2.imdecode reads dims
from the JPEG header) and YUYV (raw layout — frame size taken
from the source's :attr:`_info`, populated in ``open()``).
Sources with non-standard byte layouts override.

Returns None on decode failure (caller skips the frame).

#### MkvSource.camera_controls *(property)*

Camera-control handle (CameraControls). None for replay/mock.

#### MkvSource.mode *(property)*

CaptureMode the source was configured with. None when irrelevant.

#### MkvSource.pts_to_monotonic_ns

```python
pts_to_monotonic_ns(self, pts_us: int) -> Optional[int]
```

Convert a stream-relative `pts_us` to absolute CLOCK_MONOTONIC
nanoseconds on the source's clock domain. Returns None when the
source has no monotonic anchor (e.g. mock source pre-open).

Default impl works for any source that publishes
`t0_monotonic_ns`; concrete sources can override if they need
a non-linear mapping (none currently do).

#### MkvSource.pts_to_utc_ns

```python
pts_to_utc_ns(self, pts_us: int) -> Optional[int]
```

Convert pts_us to absolute UTC nanoseconds. Default impl uses
`t0_utc_ns`; returns None if no UTC anchor.

### CameraInfo

```python
class CameraInfo(os_id: str, camera_name: str, camera_serial: str, width: int, height: int, fps: float, is_live: bool, clock_domain: str, sku: Optional[str] = None) -> None
```

Geometry + identity descriptor for a live or replayed source.

### RawFrame

```python
class RawFrame(payload: bytes, pts_us: int, sequence: int, source_frame_idx: Optional[int] = None) -> None
```

Raw frame straight from a FrameSource — packet bytes + a single
timestamp.

``payload`` is the source's wire bytes — MJPEG packet for MJPG
captures, raw YUYV frame for YUYV captures. Suitable for either
decode (cv2.imdecode for MJPG) or fan-out to recorder (no
re-encode).

``pts_us`` is microseconds since stream start on the source's clock
domain. Source-local — integer math, no rounding. Live cameras
anchor at the V4L2 timestamp of the first emitted frame; replay
anchors at the MKV stream start (== container PTS).

Absolute time on the source's clock domain is derived as
``source.t0_monotonic_ns + pts_us * 1000`` (returns int_ns,
bit-exact, no float in the timing chain). For UTC use
``source.t0_utc_ns + pts_us * 1000``. The clock domain itself is
on :class:`CameraInfo`, not duplicated per-frame.

### resolve_source_arg

```python
resolve_source_arg(source) -> depz_camera_sdk.sources.base.FrameSource
```

Dispatch a source argument into a concrete :class:`FrameSource`.
See module docstring for recognized forms.

## Inference

### DepthInferenceBase

```python
class DepthInferenceBase()
```

Stereo disparity computation — neural, classical, or no-op.

Lifecycle (driven by :class:`DepthPipeline`):
  - construction: cheap. No model load, no matcher allocation.
  - :meth:`warmup`: heavy compile / allocation. Runs once on
    ``pipeline.start()``.
  - per-frame: pipeline calls :meth:`submit` with the rectified
    canvas pair, polls the returned :class:`InferenceFuture`, then
    calls :meth:`postprocess` on the raw output.
  - :meth:`shutdown`: release native resources. Idempotent.

Controls:
  - :meth:`controls` returns the static descriptor list — drives UI
    introspection and the default :meth:`set` clamp logic (inherited
    from :class:`~depz_camera_sdk.._controls.ControlOwner`).
  - :meth:`set(name, value)` mutates one control. Default impl
    coerces + clamps via the descriptor and stores via setattr.
    Subclasses override to add side effects (e.g. SGBM rebuilds
    its matcher when controls change) — call ``super().set(...)``
    first.

Subclasses MUST register via ``@register_inference("slug")`` —
that decorator stamps :data:`SLUG` and adds the class to the
inference registry. The class also declares its catalog
metadata (:attr:`LABEL`, :attr:`NEURAL`, :meth:`variants`) so the
UI can list it without instantiating.

#### DepthInferenceBase.variants *(classmethod)*

```python
variants(cls) -> tuple
```

Static catalog of every supported variant. Pure function —
no I/O, no environment reads, no caching needed (cheap to
recompute). Default empty; concrete inferences override.
``@register_inference`` wraps this method to stamp the
inference's slug onto each yielded :class:`Variant`.

#### DepthInferenceBase.get_variant *(classmethod)*

```python
get_variant(cls, slug: str)
```

Look up a single variant by slug. Convenience over
``next(v for v in cls.variants() if v.slug == slug)``. Raises
``KeyError`` with a helpful message listing the inference's
known variant slugs.

#### DepthInferenceBase.input_size *(property)*

``(canvas_h, canvas_w)`` — drives the calibration's canvas
remap. For neural backends this is the model's compiled input
size; for SGBM it's whatever the user picked at construction.

#### DepthInferenceBase.label *(property)*

Human-readable label for UI / logs (e.g. ``"OpenVINO CPU
(lightstereo-II)"``).

#### DepthInferenceBase.submit

```python
submit(self, canvas_l: numpy.ndarray, canvas_r: numpy.ndarray) -> depz_camera_sdk.engines.base.InferenceFuture
```

Submit one inference. Sync backends wrap the result in a
pre-completed :class:`EventFuture` so the worker's polling
loop is uniform.

#### DepthInferenceBase.postprocess

```python
postprocess(self, raw_output: numpy.ndarray, canvas_layout: depz_camera_sdk.calibration.base.CanvasLayout) -> numpy.ndarray
```

Raw inference output → 2D disparity in canvas pixels
(``valid_h × valid_w`` float32, sliced to the unpadded region
of the canvas).

#### DepthInferenceBase.warmup

```python
warmup(self) -> None
```

Pre-compile / pre-allocate. Default no-op (classical
backends usually have nothing to warm); neural backends
override to compile the engine.

#### DepthInferenceBase.abort

```python
abort(self) -> None
```

Abort the in-flight inference NOW (stop/cancel path).
Default no-op — classical backends run synchronously; neural
backends delegate to the engine (see
:class:`~depz_camera_sdk.engines.base.AbortedInference`).

#### DepthInferenceBase.shutdown

```python
shutdown(self, abandon: bool = False) -> None
```

Release native resources. Default no-op; override only when
a backend holds GPU/NPU contexts. ``abandon=True`` (stop path)
must not block on in-flight work.

#### DepthInferenceBase.set_intrinsics

```python
set_intrinsics(self, fx: float, fy: float, cx: float, cy: float) -> None
```

Push canvas-space intrinsics to the inference. Default
no-op — most backends don't need them (classical matchers,
relative-depth nets, stereo nets that already consume a
rectified pair).

Called by the pipeline from
:meth:`DepthPipeline._rebuild_calibration_geometry` whenever
the canvas remap is rebuilt (start, calibration update, mode
change). Focus-aware monocular metric-depth architectures
(UniDepth, Metric3D, DepthPro) override to remember these
and use them in their preprocess to produce metric depth.

#### DepthInferenceBase.control_get

```python
control_get(self, name: str) -> Any
```

Return the current value of one control by name — reads its
``_control_<name>`` backing attribute. Equivalent to ``host.<name>``
(the descriptor) but takes the name as a string.

#### DepthInferenceBase.control_set

```python
control_set(self, name: str, value: Any) -> Any
```

Apply one control update: look up descriptor, coerce + clamp by
``kind``/``value_type``/``min``/``max``/``step``, ``setattr``
onto ``_control_<name>``. Returns the value actually stored.

Subclasses override to add side effects (e.g. SGBM rebuilds
its matcher when controls change) — call ``super().control_set(...)``
for the coerce/store, then act on the new value.

#### DepthInferenceBase.controls_to_dict

```python
controls_to_dict(self) -> dict
```

Snapshot every current control value — for the UI live-state
pane, recorder sidecar, etc.

#### DepthInferenceBase.controls

```python
controls(self) -> tuple[depz_camera_sdk._controls.Control, ...]
```

This host's control descriptors — the class-level
:data:`_CONTROL_SPECS` tuple (same object every call, no per-call
allocation). The single public accessor shared by every
ControlOwner: filters, inference, calibration, pipeline.

### NeuralStereoInferenceBase

```python
class NeuralStereoInferenceBase(engine: depz_camera_sdk.engines.base.InferenceEngine) -> None
```

Stereo (disparity) inference backed by an ONNX/OpenVINO engine.

Per-network inferences (:class:`LightStereoInference`,
:class:`CREStereoInitialInference`, …) subclass this and define
their own ``preprocess`` / ``postprocess``. Construction takes
only a pre-built :class:`InferenceEngine`.

Engine kwargs (forwarded via ``engine_kwargs={...}`` from
:func:`build_inference`):

- ``cache_dir`` — override the SDK-wide cache root for compiled-
  model storage. Defaults to :func:`depz_camera_sdk.get_cache_dir`;
  backends that cache (OpenVINO) write into ``<cache_dir>/<backend>/``.
- ``do_models_cache`` (OpenVINO) — set False to skip caching
  entirely (useful for read-only filesystems / one-shot scripts).
- ``nireq`` (OpenVINO) — async-queue depth.

Controls: empty list — there are no runtime-tunable controls once
the model is compiled. Tune via different weights / different
engine slug instead.

#### NeuralStereoInferenceBase.architecture_slug *(property)*

Network identity slug (e.g. ``"lightstereo"``). Set by the
``@register_inference("...")`` decorator on the concrete
subclass.

#### NeuralStereoInferenceBase.engine_slug *(property)*

Engine slug (e.g. ``"openvino-CPU"``). Read from the engine
instance, which :func:`make_engine` stamps at construction.

#### NeuralStereoInferenceBase.model_path *(property)*

Resolved absolute path of the .onnx file.

#### NeuralStereoInferenceBase.input_size *(property)*

#### NeuralStereoInferenceBase.label *(property)*

#### NeuralStereoInferenceBase.warmup

```python
warmup(self) -> None
```

#### NeuralStereoInferenceBase.abort

```python
abort(self) -> None
```

#### NeuralStereoInferenceBase.shutdown

```python
shutdown(self, abandon: bool = False) -> None
```

#### NeuralStereoInferenceBase.set_intrinsics

```python
set_intrinsics(self, fx: float, fy: float, cx: float, cy: float) -> None
```

Optional hook — focus-aware mono networks (e.g. Metric3D
V2) override to remember the values for use in
:meth:`preprocess`. Default no-op.

#### NeuralStereoInferenceBase.preprocess

```python
preprocess(self, canvas_l: numpy.ndarray, canvas_r: numpy.ndarray | None) -> tuple[numpy.ndarray, ...]
```

BGR canvas pair (uint8 HxWx3) → engine input blobs.

Returns NCHW float32 blobs in the order the model declares
its inputs. Most networks return ``(left_blob, right_blob)``;
multi-stage ones (CREStereo combined) return more — e.g. 4
blobs. Mono networks ignore ``canvas_r`` (``None`` on the
mono pipeline path) and return a single-input tuple.

#### NeuralStereoInferenceBase.postprocess

```python
postprocess(self, raw_output: numpy.ndarray, canvas_layout: depz_camera_sdk.calibration.base.CanvasLayout) -> numpy.ndarray
```

Raw engine output → 2D map in canvas pixels
(``valid_h × valid_w`` float32, sliced to the unpadded region
of the canvas). Stereo subclasses return disparity; mono
subclasses return depth (or relative depth).

#### NeuralStereoInferenceBase.submit

```python
submit(self, canvas_l: numpy.ndarray, canvas_r: numpy.ndarray | None) -> depz_camera_sdk.engines.base.InferenceFuture
```

#### NeuralStereoInferenceBase.variants *(classmethod)*

```python
variants(cls) -> tuple
```

Static catalog of every supported variant. Pure function —
no I/O, no environment reads, no caching needed (cheap to
recompute). Default empty; concrete inferences override.
``@register_inference`` wraps this method to stamp the
inference's slug onto each yielded :class:`Variant`.

#### NeuralStereoInferenceBase.get_variant *(classmethod)*

```python
get_variant(cls, slug: str)
```

Look up a single variant by slug. Convenience over
``next(v for v in cls.variants() if v.slug == slug)``. Raises
``KeyError`` with a helpful message listing the inference's
known variant slugs.

#### NeuralStereoInferenceBase.control_get

```python
control_get(self, name: str) -> Any
```

Return the current value of one control by name — reads its
``_control_<name>`` backing attribute. Equivalent to ``host.<name>``
(the descriptor) but takes the name as a string.

#### NeuralStereoInferenceBase.control_set

```python
control_set(self, name: str, value: Any) -> Any
```

Apply one control update: look up descriptor, coerce + clamp by
``kind``/``value_type``/``min``/``max``/``step``, ``setattr``
onto ``_control_<name>``. Returns the value actually stored.

Subclasses override to add side effects (e.g. SGBM rebuilds
its matcher when controls change) — call ``super().control_set(...)``
for the coerce/store, then act on the new value.

#### NeuralStereoInferenceBase.controls_to_dict

```python
controls_to_dict(self) -> dict
```

Snapshot every current control value — for the UI live-state
pane, recorder sidecar, etc.

#### NeuralStereoInferenceBase.controls

```python
controls(self) -> tuple[depz_camera_sdk._controls.Control, ...]
```

This host's control descriptors — the class-level
:data:`_CONTROL_SPECS` tuple (same object every call, no per-call
allocation). The single public accessor shared by every
ControlOwner: filters, inference, calibration, pipeline.

### NeuralMonoInferenceBase

```python
class NeuralMonoInferenceBase(engine: depz_camera_sdk.engines.base.InferenceEngine) -> None
```

Monocular-depth base. Subclasses define the network-specific
preprocess + postprocess and call :meth:`_apply_depth_affine` at
the tail of postprocess.

The two controls ``depth_scale`` and ``depth_offset`` apply an affine
``scale·raw + offset``. Defaults (1.0 / 0.0) — set via
:class:`Control` after construction. Tune them once you
know one reference distance.

**Runtime controls** — read via the attribute, set via the owner's `set_*` API:

| control | type | default | range / values | unit |
|---|---|---|---|---|
| `depth_scale` | float | `1.0` | 0.001…1000.0 step 0.001 | — |
| `depth_offset` | float | `0.0` | -100.0…100.0 step 0.01 | m |

#### NeuralMonoInferenceBase.architecture_slug *(property)*

Network identity slug (e.g. ``"lightstereo"``). Set by the
``@register_inference("...")`` decorator on the concrete
subclass.

#### NeuralMonoInferenceBase.engine_slug *(property)*

Engine slug (e.g. ``"openvino-CPU"``). Read from the engine
instance, which :func:`make_engine` stamps at construction.

#### NeuralMonoInferenceBase.model_path *(property)*

Resolved absolute path of the .onnx file.

#### NeuralMonoInferenceBase.input_size *(property)*

#### NeuralMonoInferenceBase.label *(property)*

#### NeuralMonoInferenceBase.warmup

```python
warmup(self) -> None
```

#### NeuralMonoInferenceBase.abort

```python
abort(self) -> None
```

#### NeuralMonoInferenceBase.shutdown

```python
shutdown(self, abandon: bool = False) -> None
```

#### NeuralMonoInferenceBase.set_intrinsics

```python
set_intrinsics(self, fx: float, fy: float, cx: float, cy: float) -> None
```

Optional hook — focus-aware mono networks (e.g. Metric3D
V2) override to remember the values for use in
:meth:`preprocess`. Default no-op.

#### NeuralMonoInferenceBase.preprocess

```python
preprocess(self, canvas_l: numpy.ndarray, canvas_r: numpy.ndarray | None) -> tuple[numpy.ndarray, ...]
```

BGR canvas pair (uint8 HxWx3) → engine input blobs.

Returns NCHW float32 blobs in the order the model declares
its inputs. Most networks return ``(left_blob, right_blob)``;
multi-stage ones (CREStereo combined) return more — e.g. 4
blobs. Mono networks ignore ``canvas_r`` (``None`` on the
mono pipeline path) and return a single-input tuple.

#### NeuralMonoInferenceBase.postprocess

```python
postprocess(self, raw_output: numpy.ndarray, canvas_layout: depz_camera_sdk.calibration.base.CanvasLayout) -> numpy.ndarray
```

Raw engine output → 2D map in canvas pixels
(``valid_h × valid_w`` float32, sliced to the unpadded region
of the canvas). Stereo subclasses return disparity; mono
subclasses return depth (or relative depth).

#### NeuralMonoInferenceBase.submit

```python
submit(self, canvas_l: numpy.ndarray, canvas_r: numpy.ndarray | None) -> depz_camera_sdk.engines.base.InferenceFuture
```

#### NeuralMonoInferenceBase.variants *(classmethod)*

```python
variants(cls) -> tuple
```

Static catalog of every supported variant. Pure function —
no I/O, no environment reads, no caching needed (cheap to
recompute). Default empty; concrete inferences override.
``@register_inference`` wraps this method to stamp the
inference's slug onto each yielded :class:`Variant`.

#### NeuralMonoInferenceBase.get_variant *(classmethod)*

```python
get_variant(cls, slug: str)
```

Look up a single variant by slug. Convenience over
``next(v for v in cls.variants() if v.slug == slug)``. Raises
``KeyError`` with a helpful message listing the inference's
known variant slugs.

#### NeuralMonoInferenceBase.control_get

```python
control_get(self, name: str) -> Any
```

Return the current value of one control by name — reads its
``_control_<name>`` backing attribute. Equivalent to ``host.<name>``
(the descriptor) but takes the name as a string.

#### NeuralMonoInferenceBase.control_set

```python
control_set(self, name: str, value: Any) -> Any
```

Apply one control update: look up descriptor, coerce + clamp by
``kind``/``value_type``/``min``/``max``/``step``, ``setattr``
onto ``_control_<name>``. Returns the value actually stored.

Subclasses override to add side effects (e.g. SGBM rebuilds
its matcher when controls change) — call ``super().control_set(...)``
for the coerce/store, then act on the new value.

#### NeuralMonoInferenceBase.controls_to_dict

```python
controls_to_dict(self) -> dict
```

Snapshot every current control value — for the UI live-state
pane, recorder sidecar, etc.

#### NeuralMonoInferenceBase.controls

```python
controls(self) -> tuple[depz_camera_sdk._controls.Control, ...]
```

This host's control descriptors — the class-level
:data:`_CONTROL_SPECS` tuple (same object every call, no per-call
allocation). The single public accessor shared by every
ControlOwner: filters, inference, calibration, pipeline.

### SgbmInference

```python
class SgbmInference(canvas_size: 'tuple[int, int]') -> None
```

Classical Semi-Global Block Matching (cv2.StereoSGBM).

Construction takes only the canvas size; every matcher parameter
is a runtime control seeded from its descriptor default::

    inf = SgbmInference(canvas_size=(360, 640))
    # Tune controls at runtime via the pipeline:
    pipe.set_inference_control("num_disparities", 128)
    pipe.set_inference_control("block_size", 7)

**Runtime controls** — read via the attribute, set via the owner's `set_*` API:

| control | type | default | range / values | unit |
|---|---|---|---|---|
| `min_disparity` | int | `0` | -128…128 step 1 | px |
| `num_disparities` | int | `128` | 16…512 step 16 | px |
| `block_size` | int | `5` | 3…21 step 2 | px |
| `P1` | int | `0` | 0…4096 step 1 | — |
| `P2` | int | `0` | 0…16384 step 1 | — |
| `disp12_max_diff` | int | `1` | -1…32 step 1 | — |
| `uniqueness_ratio` | int | `10` | 0…50 step 1 | % |
| `speckle_window_size` | int | `100` | 0…400 step 1 | — |
| `speckle_range` | int | `2` | 0…32 step 1 | — |
| `pre_filter_cap` | int | `63` | 0…63 step 1 | — |
| `mode` | int | `2` | `0.0`, `1.0`, `2.0`, `3.0` | — |

#### SgbmInference.variants *(classmethod)*

```python
variants(cls_) -> tuple[depz_camera_sdk.inference.base.InferenceVariant, ...]
```

#### SgbmInference.label *(property)*

#### SgbmInference.input_size *(property)*

#### SgbmInference.submit

```python
submit(self, canvas_l: numpy.ndarray, canvas_r: numpy.ndarray) -> depz_camera_sdk.engines.base.InferenceFuture
```

#### SgbmInference.postprocess

```python
postprocess(self, raw_output: numpy.ndarray, canvas_layout: depz_camera_sdk.calibration.base.CanvasLayout) -> numpy.ndarray
```

#### SgbmInference.control_set

```python
control_set(self, name, value)
```

#### SgbmInference.get_variant *(classmethod)*

```python
get_variant(cls, slug: str)
```

Look up a single variant by slug. Convenience over
``next(v for v in cls.variants() if v.slug == slug)``. Raises
``KeyError`` with a helpful message listing the inference's
known variant slugs.

#### SgbmInference.warmup

```python
warmup(self) -> None
```

Pre-compile / pre-allocate. Default no-op (classical
backends usually have nothing to warm); neural backends
override to compile the engine.

#### SgbmInference.abort

```python
abort(self) -> None
```

Abort the in-flight inference NOW (stop/cancel path).
Default no-op — classical backends run synchronously; neural
backends delegate to the engine (see
:class:`~depz_camera_sdk.engines.base.AbortedInference`).

#### SgbmInference.shutdown

```python
shutdown(self, abandon: bool = False) -> None
```

Release native resources. Default no-op; override only when
a backend holds GPU/NPU contexts. ``abandon=True`` (stop path)
must not block on in-flight work.

#### SgbmInference.set_intrinsics

```python
set_intrinsics(self, fx: float, fy: float, cx: float, cy: float) -> None
```

Push canvas-space intrinsics to the inference. Default
no-op — most backends don't need them (classical matchers,
relative-depth nets, stereo nets that already consume a
rectified pair).

Called by the pipeline from
:meth:`DepthPipeline._rebuild_calibration_geometry` whenever
the canvas remap is rebuilt (start, calibration update, mode
change). Focus-aware monocular metric-depth architectures
(UniDepth, Metric3D, DepthPro) override to remember these
and use them in their preprocess to produce metric depth.

#### SgbmInference.control_get

```python
control_get(self, name: str) -> Any
```

Return the current value of one control by name — reads its
``_control_<name>`` backing attribute. Equivalent to ``host.<name>``
(the descriptor) but takes the name as a string.

#### SgbmInference.controls_to_dict

```python
controls_to_dict(self) -> dict
```

Snapshot every current control value — for the UI live-state
pane, recorder sidecar, etc.

#### SgbmInference.controls

```python
controls(self) -> tuple[depz_camera_sdk._controls.Control, ...]
```

This host's control descriptors — the class-level
:data:`_CONTROL_SPECS` tuple (same object every call, no per-call
allocation). The single public accessor shared by every
ControlOwner: filters, inference, calibration, pipeline.

### StereoBMInference

```python
class StereoBMInference(canvas_size: 'tuple[int, int]') -> None
```

Classical block matching (cv2.StereoBM).

Construction takes only the canvas size; every matcher parameter
is a runtime control seeded from its descriptor default::

    inf = StereoBMInference(canvas_size=(360, 640))
    pipe.set_inference_control("num_disparities", 96)
    pipe.set_inference_control("block_size", 9)

**Runtime controls** — read via the attribute, set via the owner's `set_*` API:

| control | type | default | range / values | unit |
|---|---|---|---|---|
| `min_disparity` | int | `0` | -128…128 step 1 | px |
| `num_disparities` | int | `96` | 16…512 step 16 | px |
| `block_size` | int | `9` | 5…51 step 2 | px |
| `pre_filter_type` | int | `1` | `0.0`, `1.0` | — |
| `pre_filter_size` | int | `9` | 5…51 step 2 | px |
| `pre_filter_cap` | int | `31` | 1…63 step 1 | — |
| `texture_threshold` | int | `10` | 0…2000 step 1 | — |
| `uniqueness_ratio` | int | `15` | 0…50 step 1 | % |
| `speckle_window_size` | int | `100` | 0…400 step 1 | — |
| `speckle_range` | int | `2` | 0…32 step 1 | — |
| `disp12_max_diff` | int | `-1` | -1…32 step 1 | — |

#### StereoBMInference.variants *(classmethod)*

```python
variants(cls_) -> tuple[depz_camera_sdk.inference.base.InferenceVariant, ...]
```

#### StereoBMInference.label *(property)*

#### StereoBMInference.input_size *(property)*

#### StereoBMInference.submit

```python
submit(self, canvas_l: numpy.ndarray, canvas_r: numpy.ndarray) -> depz_camera_sdk.engines.base.InferenceFuture
```

#### StereoBMInference.postprocess

```python
postprocess(self, raw_output: numpy.ndarray, canvas_layout: depz_camera_sdk.calibration.base.CanvasLayout) -> numpy.ndarray
```

#### StereoBMInference.control_set

```python
control_set(self, name, value)
```

#### StereoBMInference.get_variant *(classmethod)*

```python
get_variant(cls, slug: str)
```

Look up a single variant by slug. Convenience over
``next(v for v in cls.variants() if v.slug == slug)``. Raises
``KeyError`` with a helpful message listing the inference's
known variant slugs.

#### StereoBMInference.warmup

```python
warmup(self) -> None
```

Pre-compile / pre-allocate. Default no-op (classical
backends usually have nothing to warm); neural backends
override to compile the engine.

#### StereoBMInference.abort

```python
abort(self) -> None
```

Abort the in-flight inference NOW (stop/cancel path).
Default no-op — classical backends run synchronously; neural
backends delegate to the engine (see
:class:`~depz_camera_sdk.engines.base.AbortedInference`).

#### StereoBMInference.shutdown

```python
shutdown(self, abandon: bool = False) -> None
```

Release native resources. Default no-op; override only when
a backend holds GPU/NPU contexts. ``abandon=True`` (stop path)
must not block on in-flight work.

#### StereoBMInference.set_intrinsics

```python
set_intrinsics(self, fx: float, fy: float, cx: float, cy: float) -> None
```

Push canvas-space intrinsics to the inference. Default
no-op — most backends don't need them (classical matchers,
relative-depth nets, stereo nets that already consume a
rectified pair).

Called by the pipeline from
:meth:`DepthPipeline._rebuild_calibration_geometry` whenever
the canvas remap is rebuilt (start, calibration update, mode
change). Focus-aware monocular metric-depth architectures
(UniDepth, Metric3D, DepthPro) override to remember these
and use them in their preprocess to produce metric depth.

#### StereoBMInference.control_get

```python
control_get(self, name: str) -> Any
```

Return the current value of one control by name — reads its
``_control_<name>`` backing attribute. Equivalent to ``host.<name>``
(the descriptor) but takes the name as a string.

#### StereoBMInference.controls_to_dict

```python
controls_to_dict(self) -> dict
```

Snapshot every current control value — for the UI live-state
pane, recorder sidecar, etc.

#### StereoBMInference.controls

```python
controls(self) -> tuple[depz_camera_sdk._controls.Control, ...]
```

This host's control descriptors — the class-level
:data:`_CONTROL_SPECS` tuple (same object every call, no per-call
allocation). The single public accessor shared by every
ControlOwner: filters, inference, calibration, pipeline.

### RectifyOnlyInference

```python
class RectifyOnlyInference(canvas_size: 'tuple[int, int]') -> None
```

Inference-shaped no-op for pipelines that only need rectification.

Use when you want the pipeline's calibration / rectification /
canvas-fit machinery to run on every frame, but no inference. The
worker emits :class:`FramePacket` with ``rectified_left`` /
``rectified_right`` populated (per ``set_stream_enabled``) and
``disparity = depth = point_cloud = None`` on every frame. Useful
for:

- capturing rectified stereo pairs for offline analysis
- feeding rectified frames into a different downstream consumer
- sanity-checking calibration without paying inference cost

Construct with the canvas size you want — that drives the
rectifier's canvas-fit math the same way an ONNX model's input
shape would::

    inf = RectifyOnlyInference(canvas_size=(360, 640))   # H × W
    DepthPipeline(source=src, inference=inf)

The pipeline detects this class via :data:`SLUG` and routes every
frame to the photo-mode bypass path.

#### RectifyOnlyInference.variants *(classmethod)*

```python
variants(cls_) -> tuple[depz_camera_sdk.inference.base.InferenceVariant, ...]
```

#### RectifyOnlyInference.input_size *(property)*

#### RectifyOnlyInference.label *(property)*

#### RectifyOnlyInference.submit

```python
submit(self, *_args, **_kwargs) -> depz_camera_sdk.engines.base.InferenceFuture
```

#### RectifyOnlyInference.postprocess

```python
postprocess(self, raw_output: numpy.ndarray, canvas_layout: depz_camera_sdk.calibration.base.CanvasLayout) -> numpy.ndarray
```

#### RectifyOnlyInference.get_variant *(classmethod)*

```python
get_variant(cls, slug: str)
```

Look up a single variant by slug. Convenience over
``next(v for v in cls.variants() if v.slug == slug)``. Raises
``KeyError`` with a helpful message listing the inference's
known variant slugs.

#### RectifyOnlyInference.warmup

```python
warmup(self) -> None
```

Pre-compile / pre-allocate. Default no-op (classical
backends usually have nothing to warm); neural backends
override to compile the engine.

#### RectifyOnlyInference.abort

```python
abort(self) -> None
```

Abort the in-flight inference NOW (stop/cancel path).
Default no-op — classical backends run synchronously; neural
backends delegate to the engine (see
:class:`~depz_camera_sdk.engines.base.AbortedInference`).

#### RectifyOnlyInference.shutdown

```python
shutdown(self, abandon: bool = False) -> None
```

Release native resources. Default no-op; override only when
a backend holds GPU/NPU contexts. ``abandon=True`` (stop path)
must not block on in-flight work.

#### RectifyOnlyInference.set_intrinsics

```python
set_intrinsics(self, fx: float, fy: float, cx: float, cy: float) -> None
```

Push canvas-space intrinsics to the inference. Default
no-op — most backends don't need them (classical matchers,
relative-depth nets, stereo nets that already consume a
rectified pair).

Called by the pipeline from
:meth:`DepthPipeline._rebuild_calibration_geometry` whenever
the canvas remap is rebuilt (start, calibration update, mode
change). Focus-aware monocular metric-depth architectures
(UniDepth, Metric3D, DepthPro) override to remember these
and use them in their preprocess to produce metric depth.

#### RectifyOnlyInference.control_get

```python
control_get(self, name: str) -> Any
```

Return the current value of one control by name — reads its
``_control_<name>`` backing attribute. Equivalent to ``host.<name>``
(the descriptor) but takes the name as a string.

#### RectifyOnlyInference.control_set

```python
control_set(self, name: str, value: Any) -> Any
```

Apply one control update: look up descriptor, coerce + clamp by
``kind``/``value_type``/``min``/``max``/``step``, ``setattr``
onto ``_control_<name>``. Returns the value actually stored.

Subclasses override to add side effects (e.g. SGBM rebuilds
its matcher when controls change) — call ``super().control_set(...)``
for the coerce/store, then act on the new value.

#### RectifyOnlyInference.controls_to_dict

```python
controls_to_dict(self) -> dict
```

Snapshot every current control value — for the UI live-state
pane, recorder sidecar, etc.

#### RectifyOnlyInference.controls

```python
controls(self) -> tuple[depz_camera_sdk._controls.Control, ...]
```

This host's control descriptors — the class-level
:data:`_CONTROL_SPECS` tuple (same object every call, no per-call
allocation). The single public accessor shared by every
ControlOwner: filters, inference, calibration, pipeline.

### build_inference

```python
build_inference(inference_slug: str, inference_variant_slug: str, *, engine_slug: Optional[str] = None, extra_engine_kwargs: dict | None = None, extra_inference_kwargs: dict | None = None, download_progress_cb: Union[Callable[[int, int], NoneType], bool, NoneType] = None) -> depz_camera_sdk.inference.base.DepthInferenceBase
```

One-shot: ``(inference_slug, variant_slug)`` → fully-initialized
:class:`DepthInferenceBase`.

1. :func:`get_inference` resolves the inference class, then
   :meth:`get_variant` looks up the variant.
2. :meth:`InferenceVariant.resolve_files` populates the cache
   (downloads on first use; cache hit is read-only ``stat()``s).
3. ``variant.inference_kwargs`` (merged with
   ``extra_inference_kwargs``) are forwarded to the constructor.
   ``variant.engine_kwargs`` (merged with ``extra_engine_kwargs``)
   are forwarded to :func:`make_engine`.
4. Classical inferences (``cls.NEURAL = False``) are constructed
   directly. Neural ones get an :class:`InferenceEngine` built
   around the entry-point file and passed via ``engine=``.

``engine_slug`` defaults to :func:`recommend_engine` (best engine
available on this host). ``download_progress_cb`` controls the
cache-fill reporter — see :meth:`InferenceVariant.resolve_files`.

### available_inferences

```python
available_inferences() -> tuple[typing.Type[depz_camera_sdk.inference.base.DepthInferenceBase], ...]
```

Every registered inference class, sorted by slug for
deterministic UI listing.

### register_inference

```python
register_inference(slug: str)
```

Decorator: register a concrete :class:`DepthInferenceBase`
subclass under ``slug``. Stamps :attr:`SLUG`, wraps
:meth:`variants` to bind the slug onto each yielded
:class:`InferenceVariant`, and adds the class to
:data:`_INFERENCE_REGISTRY`.

### DepzStereoInference

```python
class DepzStereoInference(engine: depz_camera_sdk.engines.base.InferenceEngine) -> None
```

DEPZ Stereo — fast disparity stereo, DEPZ's own model. ImageNet-
norm preprocess, squeeze-and-crop postprocess. ONNX exports pin
(H, W) in the graph, so no INPUT_SIZE override needed.

#### DepzStereoInference.variants *(classmethod)*

```python
variants(cls_) -> tuple[depz_camera_sdk.inference.base.InferenceVariant, ...]
```

#### DepzStereoInference.preprocess

```python
preprocess(self, canvas_l: numpy.ndarray, canvas_r: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]
```

#### DepzStereoInference.postprocess

```python
postprocess(self, raw_output: numpy.ndarray, canvas_layout: depz_camera_sdk.calibration.base.CanvasLayout) -> numpy.ndarray
```

#### DepzStereoInference.architecture_slug *(property)*

Network identity slug (e.g. ``"lightstereo"``). Set by the
``@register_inference("...")`` decorator on the concrete
subclass.

#### DepzStereoInference.engine_slug *(property)*

Engine slug (e.g. ``"openvino-CPU"``). Read from the engine
instance, which :func:`make_engine` stamps at construction.

#### DepzStereoInference.model_path *(property)*

Resolved absolute path of the .onnx file.

#### DepzStereoInference.input_size *(property)*

#### DepzStereoInference.label *(property)*

#### DepzStereoInference.warmup

```python
warmup(self) -> None
```

#### DepzStereoInference.abort

```python
abort(self) -> None
```

#### DepzStereoInference.shutdown

```python
shutdown(self, abandon: bool = False) -> None
```

#### DepzStereoInference.set_intrinsics

```python
set_intrinsics(self, fx: float, fy: float, cx: float, cy: float) -> None
```

Optional hook — focus-aware mono networks (e.g. Metric3D
V2) override to remember the values for use in
:meth:`preprocess`. Default no-op.

#### DepzStereoInference.submit

```python
submit(self, canvas_l: numpy.ndarray, canvas_r: numpy.ndarray | None) -> depz_camera_sdk.engines.base.InferenceFuture
```

#### DepzStereoInference.get_variant *(classmethod)*

```python
get_variant(cls, slug: str)
```

Look up a single variant by slug. Convenience over
``next(v for v in cls.variants() if v.slug == slug)``. Raises
``KeyError`` with a helpful message listing the inference's
known variant slugs.

#### DepzStereoInference.control_get

```python
control_get(self, name: str) -> Any
```

Return the current value of one control by name — reads its
``_control_<name>`` backing attribute. Equivalent to ``host.<name>``
(the descriptor) but takes the name as a string.

#### DepzStereoInference.control_set

```python
control_set(self, name: str, value: Any) -> Any
```

Apply one control update: look up descriptor, coerce + clamp by
``kind``/``value_type``/``min``/``max``/``step``, ``setattr``
onto ``_control_<name>``. Returns the value actually stored.

Subclasses override to add side effects (e.g. SGBM rebuilds
its matcher when controls change) — call ``super().control_set(...)``
for the coerce/store, then act on the new value.

#### DepzStereoInference.controls_to_dict

```python
controls_to_dict(self) -> dict
```

Snapshot every current control value — for the UI live-state
pane, recorder sidecar, etc.

#### DepzStereoInference.controls

```python
controls(self) -> tuple[depz_camera_sdk._controls.Control, ...]
```

This host's control descriptors — the class-level
:data:`_CONTROL_SPECS` tuple (same object every call, no per-call
allocation). The single public accessor shared by every
ControlOwner: filters, inference, calibration, pipeline.

### LightStereoInference

```python
class LightStereoInference(engine: depz_camera_sdk.engines.base.InferenceEngine) -> None
```

LightStereo — fast disparity stereo. ImageNet-norm preprocess,
squeeze-and-crop postprocess. ONNX exports pin (H, W) in the
graph, so no INPUT_SIZE override needed.

#### LightStereoInference.variants *(classmethod)*

```python
variants(cls_) -> tuple[depz_camera_sdk.inference.base.InferenceVariant, ...]
```

#### LightStereoInference.preprocess

```python
preprocess(self, canvas_l: numpy.ndarray, canvas_r: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]
```

#### LightStereoInference.postprocess

```python
postprocess(self, raw_output: numpy.ndarray, canvas_layout: depz_camera_sdk.calibration.base.CanvasLayout) -> numpy.ndarray
```

#### LightStereoInference.architecture_slug *(property)*

Network identity slug (e.g. ``"lightstereo"``). Set by the
``@register_inference("...")`` decorator on the concrete
subclass.

#### LightStereoInference.engine_slug *(property)*

Engine slug (e.g. ``"openvino-CPU"``). Read from the engine
instance, which :func:`make_engine` stamps at construction.

#### LightStereoInference.model_path *(property)*

Resolved absolute path of the .onnx file.

#### LightStereoInference.input_size *(property)*

#### LightStereoInference.label *(property)*

#### LightStereoInference.warmup

```python
warmup(self) -> None
```

#### LightStereoInference.abort

```python
abort(self) -> None
```

#### LightStereoInference.shutdown

```python
shutdown(self, abandon: bool = False) -> None
```

#### LightStereoInference.set_intrinsics

```python
set_intrinsics(self, fx: float, fy: float, cx: float, cy: float) -> None
```

Optional hook — focus-aware mono networks (e.g. Metric3D
V2) override to remember the values for use in
:meth:`preprocess`. Default no-op.

#### LightStereoInference.submit

```python
submit(self, canvas_l: numpy.ndarray, canvas_r: numpy.ndarray | None) -> depz_camera_sdk.engines.base.InferenceFuture
```

#### LightStereoInference.get_variant *(classmethod)*

```python
get_variant(cls, slug: str)
```

Look up a single variant by slug. Convenience over
``next(v for v in cls.variants() if v.slug == slug)``. Raises
``KeyError`` with a helpful message listing the inference's
known variant slugs.

#### LightStereoInference.control_get

```python
control_get(self, name: str) -> Any
```

Return the current value of one control by name — reads its
``_control_<name>`` backing attribute. Equivalent to ``host.<name>``
(the descriptor) but takes the name as a string.

#### LightStereoInference.control_set

```python
control_set(self, name: str, value: Any) -> Any
```

Apply one control update: look up descriptor, coerce + clamp by
``kind``/``value_type``/``min``/``max``/``step``, ``setattr``
onto ``_control_<name>``. Returns the value actually stored.

Subclasses override to add side effects (e.g. SGBM rebuilds
its matcher when controls change) — call ``super().control_set(...)``
for the coerce/store, then act on the new value.

#### LightStereoInference.controls_to_dict

```python
controls_to_dict(self) -> dict
```

Snapshot every current control value — for the UI live-state
pane, recorder sidecar, etc.

#### LightStereoInference.controls

```python
controls(self) -> tuple[depz_camera_sdk._controls.Control, ...]
```

This host's control descriptors — the class-level
:data:`_CONTROL_SPECS` tuple (same object every call, no per-call
allocation). The single public accessor shared by every
ControlOwner: filters, inference, calibration, pipeline.

### CREStereoInitialInference

```python
class CREStereoInitialInference(engine: depz_camera_sdk.engines.base.InferenceEngine) -> None
```

High-accuracy stereo (CREStereo iter2 init export).

#### CREStereoInitialInference.variants *(classmethod)*

```python
variants(cls_) -> tuple[depz_camera_sdk.inference.base.InferenceVariant, ...]
```

#### CREStereoInitialInference.preprocess

```python
preprocess(self, canvas_l: numpy.ndarray, canvas_r: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]
```

#### CREStereoInitialInference.postprocess

```python
postprocess(self, raw_output: numpy.ndarray, canvas_layout: depz_camera_sdk.calibration.base.CanvasLayout) -> numpy.ndarray
```

#### CREStereoInitialInference.architecture_slug *(property)*

Network identity slug (e.g. ``"lightstereo"``). Set by the
``@register_inference("...")`` decorator on the concrete
subclass.

#### CREStereoInitialInference.engine_slug *(property)*

Engine slug (e.g. ``"openvino-CPU"``). Read from the engine
instance, which :func:`make_engine` stamps at construction.

#### CREStereoInitialInference.model_path *(property)*

Resolved absolute path of the .onnx file.

#### CREStereoInitialInference.input_size *(property)*

#### CREStereoInitialInference.label *(property)*

#### CREStereoInitialInference.warmup

```python
warmup(self) -> None
```

#### CREStereoInitialInference.abort

```python
abort(self) -> None
```

#### CREStereoInitialInference.shutdown

```python
shutdown(self, abandon: bool = False) -> None
```

#### CREStereoInitialInference.set_intrinsics

```python
set_intrinsics(self, fx: float, fy: float, cx: float, cy: float) -> None
```

Optional hook — focus-aware mono networks (e.g. Metric3D
V2) override to remember the values for use in
:meth:`preprocess`. Default no-op.

#### CREStereoInitialInference.submit

```python
submit(self, canvas_l: numpy.ndarray, canvas_r: numpy.ndarray | None) -> depz_camera_sdk.engines.base.InferenceFuture
```

#### CREStereoInitialInference.get_variant *(classmethod)*

```python
get_variant(cls, slug: str)
```

Look up a single variant by slug. Convenience over
``next(v for v in cls.variants() if v.slug == slug)``. Raises
``KeyError`` with a helpful message listing the inference's
known variant slugs.

#### CREStereoInitialInference.control_get

```python
control_get(self, name: str) -> Any
```

Return the current value of one control by name — reads its
``_control_<name>`` backing attribute. Equivalent to ``host.<name>``
(the descriptor) but takes the name as a string.

#### CREStereoInitialInference.control_set

```python
control_set(self, name: str, value: Any) -> Any
```

Apply one control update: look up descriptor, coerce + clamp by
``kind``/``value_type``/``min``/``max``/``step``, ``setattr``
onto ``_control_<name>``. Returns the value actually stored.

Subclasses override to add side effects (e.g. SGBM rebuilds
its matcher when controls change) — call ``super().control_set(...)``
for the coerce/store, then act on the new value.

#### CREStereoInitialInference.controls_to_dict

```python
controls_to_dict(self) -> dict
```

Snapshot every current control value — for the UI live-state
pane, recorder sidecar, etc.

#### CREStereoInitialInference.controls

```python
controls(self) -> tuple[depz_camera_sdk._controls.Control, ...]
```

This host's control descriptors — the class-level
:data:`_CONTROL_SPECS` tuple (same object every call, no per-call
allocation). The single public accessor shared by every
ControlOwner: filters, inference, calibration, pipeline.

### CREStereoCombinedInference

```python
class CREStereoCombinedInference(engine: depz_camera_sdk.engines.base.InferenceEngine) -> None
```

CREStereo combined-stage stereo (init + next, single graph).

#### CREStereoCombinedInference.variants *(classmethod)*

```python
variants(cls_) -> tuple[depz_camera_sdk.inference.base.InferenceVariant, ...]
```

#### CREStereoCombinedInference.preprocess

```python
preprocess(self, canvas_l: numpy.ndarray, canvas_r: numpy.ndarray) -> tuple[numpy.ndarray, ...]
```

#### CREStereoCombinedInference.postprocess

```python
postprocess(self, raw_output: numpy.ndarray, canvas_layout: depz_camera_sdk.calibration.base.CanvasLayout) -> numpy.ndarray
```

#### CREStereoCombinedInference.architecture_slug *(property)*

Network identity slug (e.g. ``"lightstereo"``). Set by the
``@register_inference("...")`` decorator on the concrete
subclass.

#### CREStereoCombinedInference.engine_slug *(property)*

Engine slug (e.g. ``"openvino-CPU"``). Read from the engine
instance, which :func:`make_engine` stamps at construction.

#### CREStereoCombinedInference.model_path *(property)*

Resolved absolute path of the .onnx file.

#### CREStereoCombinedInference.input_size *(property)*

#### CREStereoCombinedInference.label *(property)*

#### CREStereoCombinedInference.warmup

```python
warmup(self) -> None
```

#### CREStereoCombinedInference.abort

```python
abort(self) -> None
```

#### CREStereoCombinedInference.shutdown

```python
shutdown(self, abandon: bool = False) -> None
```

#### CREStereoCombinedInference.set_intrinsics

```python
set_intrinsics(self, fx: float, fy: float, cx: float, cy: float) -> None
```

Optional hook — focus-aware mono networks (e.g. Metric3D
V2) override to remember the values for use in
:meth:`preprocess`. Default no-op.

#### CREStereoCombinedInference.submit

```python
submit(self, canvas_l: numpy.ndarray, canvas_r: numpy.ndarray | None) -> depz_camera_sdk.engines.base.InferenceFuture
```

#### CREStereoCombinedInference.get_variant *(classmethod)*

```python
get_variant(cls, slug: str)
```

Look up a single variant by slug. Convenience over
``next(v for v in cls.variants() if v.slug == slug)``. Raises
``KeyError`` with a helpful message listing the inference's
known variant slugs.

#### CREStereoCombinedInference.control_get

```python
control_get(self, name: str) -> Any
```

Return the current value of one control by name — reads its
``_control_<name>`` backing attribute. Equivalent to ``host.<name>``
(the descriptor) but takes the name as a string.

#### CREStereoCombinedInference.control_set

```python
control_set(self, name: str, value: Any) -> Any
```

Apply one control update: look up descriptor, coerce + clamp by
``kind``/``value_type``/``min``/``max``/``step``, ``setattr``
onto ``_control_<name>``. Returns the value actually stored.

Subclasses override to add side effects (e.g. SGBM rebuilds
its matcher when controls change) — call ``super().control_set(...)``
for the coerce/store, then act on the new value.

#### CREStereoCombinedInference.controls_to_dict

```python
controls_to_dict(self) -> dict
```

Snapshot every current control value — for the UI live-state
pane, recorder sidecar, etc.

#### CREStereoCombinedInference.controls

```python
controls(self) -> tuple[depz_camera_sdk._controls.Control, ...]
```

This host's control descriptors — the class-level
:data:`_CONTROL_SPECS` tuple (same object every call, no per-call
allocation). The single public accessor shared by every
ControlOwner: filters, inference, calibration, pipeline.

### CFastFoundationStereoInference

```python
class CFastFoundationStereoInference(engine: depz_camera_sdk.engines.base.InferenceEngine) -> None
```

C-Fast-FoundationStereo — real-time zero-shot stereo disparity.
ImageNet-norm preprocess (the export strips in-graph normalization),
squeeze-and-crop postprocess. The ONNX exports pin (H, W) in the graph
and emit a single ``disparity`` output, so no INPUT_SIZE / OUTPUT_NAME
override is needed.

#### CFastFoundationStereoInference.variants *(classmethod)*

```python
variants(cls_) -> tuple[depz_camera_sdk.inference.base.InferenceVariant, ...]
```

#### CFastFoundationStereoInference.preprocess

```python
preprocess(self, canvas_l: numpy.ndarray, canvas_r: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]
```

#### CFastFoundationStereoInference.postprocess

```python
postprocess(self, raw_output: numpy.ndarray, canvas_layout: depz_camera_sdk.calibration.base.CanvasLayout) -> numpy.ndarray
```

#### CFastFoundationStereoInference.architecture_slug *(property)*

Network identity slug (e.g. ``"lightstereo"``). Set by the
``@register_inference("...")`` decorator on the concrete
subclass.

#### CFastFoundationStereoInference.engine_slug *(property)*

Engine slug (e.g. ``"openvino-CPU"``). Read from the engine
instance, which :func:`make_engine` stamps at construction.

#### CFastFoundationStereoInference.model_path *(property)*

Resolved absolute path of the .onnx file.

#### CFastFoundationStereoInference.input_size *(property)*

#### CFastFoundationStereoInference.label *(property)*

#### CFastFoundationStereoInference.warmup

```python
warmup(self) -> None
```

#### CFastFoundationStereoInference.abort

```python
abort(self) -> None
```

#### CFastFoundationStereoInference.shutdown

```python
shutdown(self, abandon: bool = False) -> None
```

#### CFastFoundationStereoInference.set_intrinsics

```python
set_intrinsics(self, fx: float, fy: float, cx: float, cy: float) -> None
```

Optional hook — focus-aware mono networks (e.g. Metric3D
V2) override to remember the values for use in
:meth:`preprocess`. Default no-op.

#### CFastFoundationStereoInference.submit

```python
submit(self, canvas_l: numpy.ndarray, canvas_r: numpy.ndarray | None) -> depz_camera_sdk.engines.base.InferenceFuture
```

#### CFastFoundationStereoInference.get_variant *(classmethod)*

```python
get_variant(cls, slug: str)
```

Look up a single variant by slug. Convenience over
``next(v for v in cls.variants() if v.slug == slug)``. Raises
``KeyError`` with a helpful message listing the inference's
known variant slugs.

#### CFastFoundationStereoInference.control_get

```python
control_get(self, name: str) -> Any
```

Return the current value of one control by name — reads its
``_control_<name>`` backing attribute. Equivalent to ``host.<name>``
(the descriptor) but takes the name as a string.

#### CFastFoundationStereoInference.control_set

```python
control_set(self, name: str, value: Any) -> Any
```

Apply one control update: look up descriptor, coerce + clamp by
``kind``/``value_type``/``min``/``max``/``step``, ``setattr``
onto ``_control_<name>``. Returns the value actually stored.

Subclasses override to add side effects (e.g. SGBM rebuilds
its matcher when controls change) — call ``super().control_set(...)``
for the coerce/store, then act on the new value.

#### CFastFoundationStereoInference.controls_to_dict

```python
controls_to_dict(self) -> dict
```

Snapshot every current control value — for the UI live-state
pane, recorder sidecar, etc.

#### CFastFoundationStereoInference.controls

```python
controls(self) -> tuple[depz_camera_sdk._controls.Control, ...]
```

This host's control descriptors — the class-level
:data:`_CONTROL_SPECS` tuple (same object every call, no per-call
allocation). The single public accessor shared by every
ControlOwner: filters, inference, calibration, pipeline.

### DepthAnythingV2SmallInference

```python
class DepthAnythingV2SmallInference(engine: depz_camera_sdk.engines.base.InferenceEngine) -> None
```

Monocular relative-depth via DPT-S backbone. ImageNet-norm
preprocess; relative-depth output → user-tunable scale + offset.

**Runtime controls** — read via the attribute, set via the owner's `set_*` API:

| control | type | default | range / values | unit |
|---|---|---|---|---|
| `depth_scale` | float | `1.0` | 0.001…1000.0 step 0.001 | — |
| `depth_offset` | float | `0.0` | -100.0…100.0 step 0.01 | m |

#### DepthAnythingV2SmallInference.variants *(classmethod)*

```python
variants(cls_) -> tuple[depz_camera_sdk.inference.base.InferenceVariant, ...]
```

#### DepthAnythingV2SmallInference.preprocess

```python
preprocess(self, canvas_l: numpy.ndarray, canvas_r: 'np.ndarray | None') -> tuple[numpy.ndarray]
```

#### DepthAnythingV2SmallInference.postprocess

```python
postprocess(self, raw_output: numpy.ndarray, canvas_layout: depz_camera_sdk.calibration.base.CanvasLayout) -> numpy.ndarray
```

#### DepthAnythingV2SmallInference.architecture_slug *(property)*

Network identity slug (e.g. ``"lightstereo"``). Set by the
``@register_inference("...")`` decorator on the concrete
subclass.

#### DepthAnythingV2SmallInference.engine_slug *(property)*

Engine slug (e.g. ``"openvino-CPU"``). Read from the engine
instance, which :func:`make_engine` stamps at construction.

#### DepthAnythingV2SmallInference.model_path *(property)*

Resolved absolute path of the .onnx file.

#### DepthAnythingV2SmallInference.input_size *(property)*

#### DepthAnythingV2SmallInference.label *(property)*

#### DepthAnythingV2SmallInference.warmup

```python
warmup(self) -> None
```

#### DepthAnythingV2SmallInference.abort

```python
abort(self) -> None
```

#### DepthAnythingV2SmallInference.shutdown

```python
shutdown(self, abandon: bool = False) -> None
```

#### DepthAnythingV2SmallInference.set_intrinsics

```python
set_intrinsics(self, fx: float, fy: float, cx: float, cy: float) -> None
```

Optional hook — focus-aware mono networks (e.g. Metric3D
V2) override to remember the values for use in
:meth:`preprocess`. Default no-op.

#### DepthAnythingV2SmallInference.submit

```python
submit(self, canvas_l: numpy.ndarray, canvas_r: numpy.ndarray | None) -> depz_camera_sdk.engines.base.InferenceFuture
```

#### DepthAnythingV2SmallInference.get_variant *(classmethod)*

```python
get_variant(cls, slug: str)
```

Look up a single variant by slug. Convenience over
``next(v for v in cls.variants() if v.slug == slug)``. Raises
``KeyError`` with a helpful message listing the inference's
known variant slugs.

#### DepthAnythingV2SmallInference.control_get

```python
control_get(self, name: str) -> Any
```

Return the current value of one control by name — reads its
``_control_<name>`` backing attribute. Equivalent to ``host.<name>``
(the descriptor) but takes the name as a string.

#### DepthAnythingV2SmallInference.control_set

```python
control_set(self, name: str, value: Any) -> Any
```

Apply one control update: look up descriptor, coerce + clamp by
``kind``/``value_type``/``min``/``max``/``step``, ``setattr``
onto ``_control_<name>``. Returns the value actually stored.

Subclasses override to add side effects (e.g. SGBM rebuilds
its matcher when controls change) — call ``super().control_set(...)``
for the coerce/store, then act on the new value.

#### DepthAnythingV2SmallInference.controls_to_dict

```python
controls_to_dict(self) -> dict
```

Snapshot every current control value — for the UI live-state
pane, recorder sidecar, etc.

#### DepthAnythingV2SmallInference.controls

```python
controls(self) -> tuple[depz_camera_sdk._controls.Control, ...]
```

This host's control descriptors — the class-level
:data:`_CONTROL_SPECS` tuple (same object every call, no per-call
allocation). The single public accessor shared by every
ControlOwner: filters, inference, calibration, pipeline.

### Metric3DV2VitSmallInference

```python
class Metric3DV2VitSmallInference(engine: depz_camera_sdk.engines.base.InferenceEngine) -> None
```

Monocular metric depth with focal-length awareness (Metric3D V2,
ViT-S/14 backbone).

**Runtime controls** — read via the attribute, set via the owner's `set_*` API:

| control | type | default | range / values | unit |
|---|---|---|---|---|
| `depth_scale` | float | `1.0` | 0.001…1000.0 step 0.001 | — |
| `depth_offset` | float | `0.0` | -100.0…100.0 step 0.01 | m |

#### Metric3DV2VitSmallInference.variants *(classmethod)*

```python
variants(cls_) -> tuple[depz_camera_sdk.inference.base.InferenceVariant, ...]
```

#### Metric3DV2VitSmallInference.set_intrinsics

```python
set_intrinsics(self, fx: float, fy: float, cx: float, cy: float) -> None
```

#### Metric3DV2VitSmallInference.preprocess

```python
preprocess(self, canvas_l: numpy.ndarray, canvas_r: 'np.ndarray | None') -> tuple[numpy.ndarray]
```

#### Metric3DV2VitSmallInference.postprocess

```python
postprocess(self, raw_output: numpy.ndarray, canvas_layout: depz_camera_sdk.calibration.base.CanvasLayout) -> numpy.ndarray
```

#### Metric3DV2VitSmallInference.architecture_slug *(property)*

Network identity slug (e.g. ``"lightstereo"``). Set by the
``@register_inference("...")`` decorator on the concrete
subclass.

#### Metric3DV2VitSmallInference.engine_slug *(property)*

Engine slug (e.g. ``"openvino-CPU"``). Read from the engine
instance, which :func:`make_engine` stamps at construction.

#### Metric3DV2VitSmallInference.model_path *(property)*

Resolved absolute path of the .onnx file.

#### Metric3DV2VitSmallInference.input_size *(property)*

#### Metric3DV2VitSmallInference.label *(property)*

#### Metric3DV2VitSmallInference.warmup

```python
warmup(self) -> None
```

#### Metric3DV2VitSmallInference.abort

```python
abort(self) -> None
```

#### Metric3DV2VitSmallInference.shutdown

```python
shutdown(self, abandon: bool = False) -> None
```

#### Metric3DV2VitSmallInference.submit

```python
submit(self, canvas_l: numpy.ndarray, canvas_r: numpy.ndarray | None) -> depz_camera_sdk.engines.base.InferenceFuture
```

#### Metric3DV2VitSmallInference.get_variant *(classmethod)*

```python
get_variant(cls, slug: str)
```

Look up a single variant by slug. Convenience over
``next(v for v in cls.variants() if v.slug == slug)``. Raises
``KeyError`` with a helpful message listing the inference's
known variant slugs.

#### Metric3DV2VitSmallInference.control_get

```python
control_get(self, name: str) -> Any
```

Return the current value of one control by name — reads its
``_control_<name>`` backing attribute. Equivalent to ``host.<name>``
(the descriptor) but takes the name as a string.

#### Metric3DV2VitSmallInference.control_set

```python
control_set(self, name: str, value: Any) -> Any
```

Apply one control update: look up descriptor, coerce + clamp by
``kind``/``value_type``/``min``/``max``/``step``, ``setattr``
onto ``_control_<name>``. Returns the value actually stored.

Subclasses override to add side effects (e.g. SGBM rebuilds
its matcher when controls change) — call ``super().control_set(...)``
for the coerce/store, then act on the new value.

#### Metric3DV2VitSmallInference.controls_to_dict

```python
controls_to_dict(self) -> dict
```

Snapshot every current control value — for the UI live-state
pane, recorder sidecar, etc.

#### Metric3DV2VitSmallInference.controls

```python
controls(self) -> tuple[depz_camera_sdk._controls.Control, ...]
```

This host's control descriptors — the class-level
:data:`_CONTROL_SPECS` tuple (same object every call, no per-call
allocation). The single public accessor shared by every
ControlOwner: filters, inference, calibration, pipeline.

### InferenceVariant

```python
class InferenceVariant(slug: str, label: str, files_to_download: tuple[depz_camera_sdk.inference.base.ModelFile, ...] = (), inference_kwargs: dict[str, typing.Any] = <factory>, engine_kwargs: dict[str, typing.Any] = <factory>, configurations: dict[str, typing.Any] = <factory>, _inference_slug: str = '') -> None
```

One concrete configuration of an inference.

``slug`` is the second identifier in the (inference, variant)
pair — stable, URL-safe, unique within the inference's catalog.
``label`` is the human-facing string the UI shows in the dropdown.

``files_to_download`` lists every file the variant needs on disk.
Empty tuple = nothing to fetch (classical algorithms — SGBM,
StereoBM, rectify-only). For neural variants the **first** entry
is the entry point the engine loads.

``inference_kwargs`` are forwarded verbatim to the inference's
constructor by :func:`build_inference`
(``cls(engine=..., **variant.inference_kwargs)``). Use this for
per-variant ctor args — SGBM/StereoBM/RectifyOnly ``canvas_size``,
etc.

``engine_kwargs`` are forwarded to :func:`make_engine` when the
engine is built (``make_engine(slug, model_path, **engine_kwargs)``).
Use this for per-variant engine controls (``input_shape`` for
dynamic-shape ONNXes, ``providers``, ``cache_dir``).

``configurations`` is opaque to the framework — free-form metadata for the
UI / debugging (``size``, ``dataset``, ``input_size``, …). Not
passed to either the constructor or the engine.

``inference_slug`` is filled in by ``register_inference`` when the
inference enumerates its variants — concrete subclasses don't pass
it when constructing the dataclass.

#### InferenceVariant.inference_slug *(property)*

Inference identifier (e.g. ``"lightstereo"``). Raises
``ValueError`` when read on a variant whose registry binding
hasn't happened (``_inference_slug`` is empty).

#### InferenceVariant.cache_dir *(property)*

Deterministic cache directory for this variant. Read-only
path computation — does NOT create the directory. Layout:
``<cache_root>/<inference_slug>/<slug>/``.

#### InferenceVariant.inference *(property)*

The inference class this variant belongs to — looked up
via :func:`get_inference` from :attr:`inference_slug`.

#### InferenceVariant.is_cached *(property)*

True iff every file in :attr:`files_to_download` is on disk
with matching size + sha256 (when declared).

#### InferenceVariant.resolve_files

```python
resolve_files(self, *, download_progress_cb: Union[Callable[[int, int], NoneType], bool, NoneType] = None) -> list[pathlib.Path]
```

Ensure the variant's cache directory is populated, then
return absolute paths of every file in
:attr:`files_to_download` (entry-point first).

### ModelFile

```python
class ModelFile(slug: str, filename: str, sha256: Optional[str] = None, size_bytes: Optional[int] = None, archive_member: Optional[str] = None) -> None
```

One file an :class:`InferenceVariant` needs at runtime.

``slug`` is the model-registry identifier; the SDK resolves it
to a public download URL at fetch time via the catalog service.
Slugs are stable — moving a model between mirrors / CDNs / buckets
is a server-side edit, not a client-side code change.

``filename`` is the on-disk basename inside the variant cache
directory. For archive downloads (``archive_member`` set), this
is the *extracted* filename — the archive itself is discarded.

``sha256`` is optional but strongly recommended — it's the only
way :meth:`InferenceVariant.resolve_files` can tell a corrupted-
but-right-size file from a good one. ``size_bytes`` is advisory:
surfaced to the UI for download-progress totals before the HTTP
response is read.

``archive_member`` extracts a single file from an archive payload
(zip currently supported). The resolved URL points at an archive,
we open it in-memory / via tempfile, copy the named member into
``filename``, and discard the archive — variant cache stays small.

#### ModelFile.is_cached

```python
is_cached(self, target: pathlib.Path) -> bool
```

True iff ``target`` exists with matching size + sha256
(when declared). Read-only — never opens a socket. On the hot
read path we don't recompute sha256: it's verified at download
time, and a corrupted-but-right-size file is rare enough that
it surfaces as an ONNX-load error one layer up.

#### ModelFile.resolve

```python
resolve(self, target: pathlib.Path, download_progress_cb: Optional[Callable[[int, int], NoneType]] = None) -> None
```

Ensure the file is on disk at ``target``: no-op if
:meth:`is_cached`, otherwise :meth:`download`.

#### ModelFile.download

```python
download(self, target: pathlib.Path, download_progress_cb: Optional[Callable[[int, int], NoneType]] = None) -> None
```

Resolve :attr:`slug` to a URL, fetch it, and land the result
at ``target`` atomically. Direct file download by default;
archive-extract flow when :attr:`archive_member` is set.

### get_inference

```python
get_inference(slug: str) -> Type[depz_camera_sdk.inference.base.DepthInferenceBase]
```

Look up an inference class by slug. Raises ``KeyError`` with a
friendly message listing the registered slugs if the slug is
unknown.

### cache_root

```python
cache_root() -> pathlib.Path
```

Root directory for downloaded variant cache. Sub-directory of
the SDK-wide cache root (:func:`depz_camera_sdk.get_cache_dir`), so
every consumer benefits from the same redirect. Created lazily by
:meth:`InferenceVariant.resolve_files` when the first variant lands.

## Engines

### available_engines

```python
available_engines() -> list[str]
```

Alias of ``list_engines("available")`` — the host's usable
engines, priority-sorted. Kept as the short name for the
overwhelmingly common case; see :func:`list_engines` for the other
scopes.

### list_engines

```python
list_engines(scope: str = 'available') -> list[str]
```

Engine slugs for the requested `scope`, **sorted by priority**.

scope="available" (default)
    Loadable on this host right now, minus policy-hidden engines —
    what UIs should offer and recommenders should pick from.
    ``[0]`` is the best choice (== :func:`recommend_engine`).
scope="detected"
    Loadable on this host *including* hidden engines — for
    diagnostics ("what did the probe actually see?").
scope="full"
    Every (kind, device/EP) combination the SDK supports, straight
    from :data:`_PRIORITY_ORDER` — static, no host probing. Includes
    hidden entries; a slug's absence here means the SDK has no code
    path for it at all.
scope="hidden"
    The policy-hidden slugs (see :data:`_HIDDEN_ENGINES`) — what
    separates "detected" from "available".

### recommend_engine

```python
recommend_engine() -> Optional[str]
```

Best available engine slug for this host, or None if none loadable.

### InferenceEngine

```python
class InferenceEngine(model_path: str) -> None
```

Stereo inference backend.

Lifecycle:
    engine = SomeEngine(model_path, ...)
    engine.warmup()
    future = engine.submit(*blobs)
    ...
    if future.done(): disp = future.result()
    ...
    engine.shutdown()

Most stereo nets have 2 image inputs (L, R). Multi-stage variants
(e.g. CREStereo "combined" with init/next pairs at two resolutions)
declare 4 inputs. The architecture's `preprocess` produces blobs in
the order the model declares them; the engine just forwards them.

Thread-safety:
    - submit() / shutdown() called from the pipeline worker thread only.
    - InferenceFuture.done()/result() may be called from the same
      worker — no other thread should touch them.

#### InferenceEngine.input_shapes *(property)*

Per-input NCHW shapes in declared model order. The single
engine-introspection primitive: warmup uses it to allocate
zero-filled blobs, and :attr:`input_size` derives the canvas
H×W from it.

#### InferenceEngine.input_size *(property)*

(canvas_h, canvas_w) of the *largest* model input — the
architecture's canvas size. Multi-resolution models report
the full-res input; downsampled inputs are produced by the
architecture's preprocess from the same canvas.

#### InferenceEngine.submit

```python
submit(self, *blobs: numpy.ndarray) -> depz_camera_sdk.engines.base.InferenceFuture
```

Submit one inference. The number/shape of blobs must match
:attr:`input_shapes` in declared order.

#### InferenceEngine.abort

```python
abort(self) -> None
```

Abort the in-flight inference NOW (milliseconds, not minutes).
Terminal for the engine instance — only the stop/cancel path
calls it, right before :meth:`shutdown`. The aborted future
fails with :class:`AbortedInference`. Default no-op (classical
backends run synchronously and have nothing in flight).

#### InferenceEngine.shutdown

```python
shutdown(self, abandon: bool = False) -> None
```

Release resources. ``abandon=True`` (the stop path, after
:meth:`abort`) must NOT block on in-flight work — the process
is about to exit and any straggler native thread dies with it.

#### InferenceEngine.warmup

```python
warmup(self) -> None
```

Pre-compile + first inference. Submits one all-zeros tensor
per declared input. GPU/NPU first-compile can take 30–60 s;
OpenVINO caches the result on disk so subsequent loads are
sub-second. Override only if a backend has a cheaper warmup.

### InferenceFuture

```python
class InferenceFuture()
```

Result handle for one in-flight inference.

#### InferenceFuture.done

```python
done(self) -> bool
```

#### InferenceFuture.wait

```python
wait(self, timeout: Optional[float] = None) -> bool
```

Block up to `timeout` seconds. Return True if completed.

#### InferenceFuture.result

```python
result(self) -> numpy.ndarray
```

Return raw disparity output (h_canvas × w_canvas float32).
Raises if errored or not yet done.

### EventFuture

```python
class EventFuture() -> None
```

Generic Future for engines that fire a callback on completion
(OpenVINO AsyncInferQueue, ThreadPoolExecutor.submit's wrapper).

Two-sided contract:
- **Consumer side** (pipeline) sees only the
  :class:`InferenceFuture` interface (`done` / `wait` / `result`).
- **Engine side** (subclass / engine implementer) calls exactly one
  of :meth:`set_result` / :meth:`set_error` when inference
  completes. Both are idempotent in the sense that the second call
  is a no-op (once the event is set the future is done).

#### EventFuture.done

```python
done(self) -> bool
```

#### EventFuture.wait

```python
wait(self, timeout: Optional[float] = None) -> bool
```

#### EventFuture.result

```python
result(self) -> numpy.ndarray
```

#### EventFuture.set_result

```python
set_result(self, result: numpy.ndarray) -> None
```

Mark the future complete with a result. Engine-side.

#### EventFuture.set_error

```python
set_error(self, exc: BaseException) -> None
```

Mark the future failed. Engine-side.

### ENGINE_CLASSES

`tuple` constant.

Built-in immutable sequence.

If no argument is given, the constructor returns an empty tuple.
If iterable is specified the tuple is initialized from iterable's items.

If the argument is a tuple, the return value is the same object.

### class_rank

```python
class_rank(klass: str) -> int
```

Sort key for engine classes — higher = more capable. Raises
:class:`ValueError` on a class not in :data:`ENGINE_CLASSES` (bad
model metadata should fail loudly, not sort arbitrarily).

### classify_engine

```python
classify_engine(slug: str) -> str
```

Coarse hardware class (see :data:`ENGINE_CLASSES`) for an engine
slug. OpenVINO multi-device suffixes (``GPU.0``) strip before
lookup; unknown slugs default to ``"cpu"`` — the conservative floor
for capability matching.

### make_engine

```python
make_engine(slug: str, model_path: str, **kwargs) -> depz_camera_sdk.engines.base.InferenceEngine
```

Construct the engine identified by `slug` against `model_path`.
`slug` is one of the strings returned by :func:`available_engines`,
e.g. ``"openvino-CPU"`` or ``"onnx-cuda"``. User input is accepted
in any case — both kind and device/EP tail are normalized to the
canonical form each backend expects (OpenVINO devices uppercase,
ONNX EPs lowercase).

Extra kwargs are passed through to the backend constructor.
The most useful one is ``cache_dir=`` — backends that cache
compiled models on disk (currently OpenVINO) will use
``<cache_dir>/<backend>/`` as the cache root. Defaults to
:func:`depz_camera_sdk.get_cache_dir` when omitted.

### parse_engine_slug

```python
parse_engine_slug(slug: str) -> tuple[str, str]
```

Split an engine slug into (kind, device_or_ep). The kind is
lowercased — user input is accepted in any case; the tail is
returned verbatim and normalized later by :func:`make_engine`
per-kind. Raises :class:`ValueError` if the slug doesn't have
the expected `kind-tail` shape.

>>> parse_engine_slug("openvino-CPU")
('openvino', 'CPU')
>>> parse_engine_slug("onnx-cuda")
('onnx', 'cuda')
>>> parse_engine_slug("OPENVINO-CPU")
('openvino', 'CPU')
>>> parse_engine_slug("bogus")
Traceback (most recent call last):
    ...
ValueError: engine slug must look like '<kind>-<device_or_ep>', got 'bogus'

### priority_rank

```python
priority_rank(slug: str) -> int
```

Public alias of :func:`_priority_rank` — the canonical engine
priority for a slug. Exposed so downstream consumers (notably
``depz_camera_viewer.model_selector``) can rank engines without
duplicating :data:`_PRIORITY_ORDER` (and the ARM swap). Returns are
sort keys (lower = higher priority); only relative order matters.

## Calibration

### Calibration

```python
class Calibration(*, canvas_fit_mode: 'str | None' = None) -> None
```

Calibration: rectifier + canvas-fit remap + depth/cloud projection.

Supports both stereo (two-camera) and mono (single-camera) wire
payloads — see :attr:`is_stereo`. On the mono path the right-eye
fields and extrinsics stay ``None`` and the depth projection is
expected to come from a monocular depth inference rather than from
:meth:`disparity_to_depth`.

Concrete subclasses register a stable :attr:`POLYMORPHIC_NAME`
(the cereal lens-model identifier) and declare their own
lens-specific controls (``rectify_alpha``, FoVs, ...) as typed control
class attributes. The shared :attr:`canvas_fit_mode` control lives
here on the base, so every subclass inherits it.

Runtime params follow the same descriptor pattern as
:class:`~depz_camera_sdk.filters.base.Filter` and
:class:`~depz_camera_sdk.inference.base.DepthInferenceBase` — each is a
:class:`~depz_camera_sdk._controls.Control` (with kind / min / max / step /
enum_values / unit / label). :meth:`control_set` (ControlOwner contract,
overridden here) coerces + stores via the descriptor and triggers
:meth:`_compute_rectification` (or canvas remap invalidation, for
``canvas_fit_mode``). Subclasses override only when they need
sentinel handling or other special semantics.

All subclasses share the persistence shape, dispatch, canvas-fit
math, and depth-projection math — that all lives on this base.

**Runtime controls** — read via the attribute, set via the owner's `set_*` API:

| control | type | default | range / values | unit |
|---|---|---|---|---|
| `canvas_fit_mode` | str | `fit-with-padding-middle` | `fit-with-padding-middle`, `fit-with-padding-upper-left`, `fill-with-cropping` | — |

#### Calibration.SDK_PARAMS *(property)*

Names of every runtime control. Derived from :meth:`controls`
— kept as a property (not ClassVar any more) for backwards
compatibility with callers that snapshot the namespace.

#### Calibration.is_loaded *(property)*

#### Calibration.baseline_m *(property)*

#### Calibration.focal_native_px *(property)*

#### Calibration.is_stereo *(property)*

True when the calibration was built from two cameras (left +
right, baseline + transform). False for a single-camera (mono)
calibration — no right-eye intrinsics, no baseline. Mono
pipelines route depth from the inference instead of computing
it via :meth:`disparity_to_depth`.

#### Calibration.canvas_layout *(property)*

Geometry of the rectified eye inside the canvas. ``None``
until :meth:`compute_canvas_remap` runs; reset to ``None`` when
rectification params change.

#### Calibration.P_left_canvas *(property)*

Canvas-space projection matrix of the left rectified eye —
a 3×4 ``[K_canvas | t]`` (intrinsics in canvas pixels, plus
baseline-translation column for stereo / zeros for mono).

``None`` until :meth:`compute_canvas_remap` runs. Read by
focus-aware mono inference backends (UniDepth, Metric3D) so
they can ingest the right intrinsics for the canvas the
pipeline asks them to consume.

#### Calibration.can_handle *(classmethod)*

```python
can_handle(cls, d: dict) -> bool
```

True when any camera in the dict advertises this class's
:attr:`POLYMORPHIC_NAME`. That's the sole identification signal —
no ``kind`` tag, no structural sniffing.

#### Calibration.from_dict *(classmethod)*

```python
from_dict(cls, d: dict) -> 'Calibration'
```

Build a ``Calibration`` from a cereal-shaped dict.

- Called on the abstract base (``Calibration.from_dict(d)``):
  dispatches to the registered impl whose :meth:`can_handle`
  returns True (first match wins).
- Called on a concrete subclass: directly constructs an
  instance via :meth:`_populate` + :meth:`_compute_rectification`.

#### Calibration.load

```python
load(self, data: dict) -> None
```

Populate from a cereal-shaped dict and recompute rectification.
Convenience wrapper for callers that already have a constructed
instance (e.g. a subclass picked up from config) and want to
mutate it in place rather than building a new one via
:meth:`from_dict`.

#### Calibration.to_dict

```python
to_dict(self) -> dict
```

Emit the standard cereal-shaped dict. Subclasses don't
usually need to override — only :meth:`_distortion_params_dict`.

Mono calibrations emit a single-camera entry with no ``role``
key (`role=None`) and `camera_index=0`. Stereo emits the
``role="left"`` / ``role="right"`` pair with indices 0 / 1.

#### Calibration.apply_update

```python
apply_update(self, patch: dict) -> None
```

Apply an online-recalibrator patch in-place.

Two forms supported:
  • Full-state — patch contains a cereal-shaped ``cameras`` list
    (anywhere via :meth:`_extract_cameras`). Re-populate
    everything and recompute rectification.
  • Partial — class-specific subset; dispatched to
    :meth:`_apply_partial`.

Pipeline drains in-flight inference + resets stateful filters
BEFORE calling this, so impls can rebuild eagerly without locks.

#### Calibration.adapt_to_resolution

```python
adapt_to_resolution(self, eye_w: int, eye_h: int) -> None
```

Adjust intrinsics so they match the actual capture eye-size.

Default policy: if the capture size is a uniform scale of the
calibration size, rescale K/image_size by that factor; otherwise
raise. ``D`` (distortion coefficients) is scale-invariant under
uniform image rescale for both pinhole-radtan and Kannala–Brandt.

Subclasses override only when they need different behaviour
(e.g. an image-size-invariant calibration that no-ops here).

#### Calibration.control_set

```python
control_set(self, key: str, value: Any) -> Any
```

Update one named runtime control and rebuild geometry. Same
side-effect-on-control_set pattern as :class:`Filter` /
:class:`DepthInferenceBase` — base impl coerces and stores, this
override layers on the geometry rebuild.

Side-effect dispatch: :meth:`_compute_rectification` for numeric
controls, canvas-state invalidation for ``canvas_fit_mode`` (caller
rebuilds the remap separately, since canvas size also varies).

Subclasses override only when they need special semantics
(sentinels, cross-control constraints) — call ``super().control_set``
for the unaffected branches.

#### Calibration.set_canvas_fit_mode

```python
set_canvas_fit_mode(self, mode: str) -> None
```

#### Calibration.compute_canvas_remap

```python
compute_canvas_remap(self, canvas_w: int, canvas_h: int) -> None
```

Build a combined rectify + canvas-fit remap (cv2.remap from raw
eye directly to canvas-sized rectified image, in one pass).

Generic w.r.t. concrete calibration kind: uses the K/D/R/P
produced by `_compute_rectification` and dispatches the actual
kernel via `_init_undistort_rectify_map`. On the mono path
only the left maps are built; the right buffer stays unallocated
and :meth:`canvas_remap` returns ``(canvas_l, None)``.

#### Calibration.canvas_remap

```python
canvas_remap(self, raw_l: numpy.ndarray, raw_r: Optional[numpy.ndarray]) -> tuple[numpy.ndarray, typing.Optional[numpy.ndarray]]
```

Rectify + scale-to-canvas in one cv2.remap pass per eye.

┌──────────────────────────────────────────────────────────┐
│ DANGER — ALIASED RETURN                                  │
│ The returned arrays are views into internal pre-          │
│ allocated buffers. They become INVALID on the next call  │
│ to canvas_remap (the buffers are reused). Lifetime  │
│ ends when:                                               │
│   • the next canvas_remap call starts, OR           │
│   • compute_canvas_remap reallocates buffers (resize), OR│
│   • the Calibration instance is GC'd.                    │
│ If you need to retain past any of those, use             │
│ `canvas_remap_copy` instead, or `.copy()` the views │
│ explicitly. Anything that holds the views across worker  │
│ iterations is a silent corruption bug.                   │
└──────────────────────────────────────────────────────────┘

Pipeline's worker is the only caller that uses this safely —
it consumes the views inside the same iteration (model
preprocess copies into a fresh tensor; rect_l/rect_r get
explicit `.copy()` before going into the inflight slot).

#### Calibration.canvas_remap_copy

```python
canvas_remap_copy(self, raw_l: numpy.ndarray, raw_r: Optional[numpy.ndarray]) -> tuple[numpy.ndarray, typing.Optional[numpy.ndarray]]
```

Safe variant of `canvas_remap` — returns fresh
copies that have no lifetime constraint.

Use this when you don't fully understand or can't enforce
the aliasing contract documented on `canvas_remap`.
Costs one extra allocation pair per call (~3 MB at
canvas 1280×720); negligible vs the cv2.remap itself for
correctness-critical or low-fps consumers.

#### Calibration.rectify

```python
rectify(self, left: numpy.ndarray, right: Optional[numpy.ndarray] = None) -> tuple[numpy.ndarray, typing.Optional[numpy.ndarray]]
```

Native-res rectified pair. Fresh arrays each call.

Diagnostic / API-parity utility. The pipeline worker uses
:meth:`canvas_remap` instead — it fuses rectify + canvas-fit
into one cv2.remap. On mono calibrations (or when ``right`` is
None) returns ``(rectified_left, None)``.

#### Calibration.disparity_to_depth

```python
disparity_to_depth(self, disparity_canvas_px: numpy.ndarray) -> numpy.ndarray
```

Convert canvas-pixel disparity to metric depth.

Returns a fresh array each call (FramePacket.depth is held in
the output queue; reusing a buffer would alias multiple packets).
Default: pinhole formula `Z = baseline · focal / disparity`.
Override for non-pinhole models. Raises on mono calibrations
(depth comes from a monocular depth inference instead).

#### Calibration.depth_to_point_cloud

```python
depth_to_point_cloud(self, depth_m: numpy.ndarray) -> numpy.ndarray
```

Back-project depth to a 3D point cloud (h, w, 3) in meters.

Default: pinhole unprojection using rectified P_left intrinsics.
Override for fisheye / omnidir.

#### Calibration.control_get

```python
control_get(self, name: str) -> Any
```

Return the current value of one control by name — reads its
``_control_<name>`` backing attribute. Equivalent to ``host.<name>``
(the descriptor) but takes the name as a string.

#### Calibration.controls_to_dict

```python
controls_to_dict(self) -> dict
```

Snapshot every current control value — for the UI live-state
pane, recorder sidecar, etc.

#### Calibration.controls

```python
controls(self) -> tuple[depz_camera_sdk._controls.Control, ...]
```

This host's control descriptors — the class-level
:data:`_CONTROL_SPECS` tuple (same object every call, no per-call
allocation). The single public accessor shared by every
ControlOwner: filters, inference, calibration, pipeline.

### CanvasLayout

```python
class CanvasLayout(scale: float, valid_w: int, valid_h: int, pad_x: int, pad_y: int, eye_h: int, eye_w: int) -> None
```

Geometry of the rectified eye inside the model's fixed-size canvas.

Produced by :meth:`Calibration.compute_canvas_remap`. The model
sees a `canvas_w × canvas_h` image, but only ``valid_w × valid_h``
of it is real rectified content; the rest is letterbox padding (or,
in ``crop`` mode, off-canvas content the rectifier didn't draw).
Postprocess slices the disparity back to that valid region so
downstream stages don't see padding.

### OpenCVBasicCalibration

```python
class OpenCVBasicCalibration(*, rectify_alpha: 'float | None' = None, canvas_fit_mode: 'str | None' = None) -> None
```

Pinhole + 8-coefficient radtan distortion.

Intrinsics: K = [[f, 0, cx], [0, f·ar, cy], [0, 0, 1]]
            D = [k1, k2, p1, p2, k3, k4, k5, k6]   (cv2 ordering)
Extrinsics: rotation as Rodrigues vector (rx, ry, rz), translation
            in meters (x, y, z), measured for the RIGHT camera in
            the LEFT camera's frame.
Rectification: ``cv2.stereoRectify`` + ``cv2.initUndistortRectifyMap``.

**Runtime controls** — read via the attribute, set via the owner's `set_*` API:

| control | type | default | range / values | unit |
|---|---|---|---|---|
| `canvas_fit_mode` | str | `fit-with-padding-middle` | `fit-with-padding-middle`, `fit-with-padding-upper-left`, `fill-with-cropping` | — |
| `rectify_alpha` | float | `0.0` | 0.0…1.0 step 0.01 | — |

#### OpenCVBasicCalibration.control_set

```python
control_set(self, key: str, value: Any) -> Any
```

#### OpenCVBasicCalibration.SDK_PARAMS *(property)*

Names of every runtime control. Derived from :meth:`controls`
— kept as a property (not ClassVar any more) for backwards
compatibility with callers that snapshot the namespace.

#### OpenCVBasicCalibration.is_loaded *(property)*

#### OpenCVBasicCalibration.baseline_m *(property)*

#### OpenCVBasicCalibration.focal_native_px *(property)*

#### OpenCVBasicCalibration.is_stereo *(property)*

True when the calibration was built from two cameras (left +
right, baseline + transform). False for a single-camera (mono)
calibration — no right-eye intrinsics, no baseline. Mono
pipelines route depth from the inference instead of computing
it via :meth:`disparity_to_depth`.

#### OpenCVBasicCalibration.canvas_layout *(property)*

Geometry of the rectified eye inside the canvas. ``None``
until :meth:`compute_canvas_remap` runs; reset to ``None`` when
rectification params change.

#### OpenCVBasicCalibration.P_left_canvas *(property)*

Canvas-space projection matrix of the left rectified eye —
a 3×4 ``[K_canvas | t]`` (intrinsics in canvas pixels, plus
baseline-translation column for stereo / zeros for mono).

``None`` until :meth:`compute_canvas_remap` runs. Read by
focus-aware mono inference backends (UniDepth, Metric3D) so
they can ingest the right intrinsics for the canvas the
pipeline asks them to consume.

#### OpenCVBasicCalibration.can_handle *(classmethod)*

```python
can_handle(cls, d: dict) -> bool
```

True when any camera in the dict advertises this class's
:attr:`POLYMORPHIC_NAME`. That's the sole identification signal —
no ``kind`` tag, no structural sniffing.

#### OpenCVBasicCalibration.from_dict *(classmethod)*

```python
from_dict(cls, d: dict) -> 'Calibration'
```

Build a ``Calibration`` from a cereal-shaped dict.

- Called on the abstract base (``Calibration.from_dict(d)``):
  dispatches to the registered impl whose :meth:`can_handle`
  returns True (first match wins).
- Called on a concrete subclass: directly constructs an
  instance via :meth:`_populate` + :meth:`_compute_rectification`.

#### OpenCVBasicCalibration.load

```python
load(self, data: dict) -> None
```

Populate from a cereal-shaped dict and recompute rectification.
Convenience wrapper for callers that already have a constructed
instance (e.g. a subclass picked up from config) and want to
mutate it in place rather than building a new one via
:meth:`from_dict`.

#### OpenCVBasicCalibration.to_dict

```python
to_dict(self) -> dict
```

Emit the standard cereal-shaped dict. Subclasses don't
usually need to override — only :meth:`_distortion_params_dict`.

Mono calibrations emit a single-camera entry with no ``role``
key (`role=None`) and `camera_index=0`. Stereo emits the
``role="left"`` / ``role="right"`` pair with indices 0 / 1.

#### OpenCVBasicCalibration.apply_update

```python
apply_update(self, patch: dict) -> None
```

Apply an online-recalibrator patch in-place.

Two forms supported:
  • Full-state — patch contains a cereal-shaped ``cameras`` list
    (anywhere via :meth:`_extract_cameras`). Re-populate
    everything and recompute rectification.
  • Partial — class-specific subset; dispatched to
    :meth:`_apply_partial`.

Pipeline drains in-flight inference + resets stateful filters
BEFORE calling this, so impls can rebuild eagerly without locks.

#### OpenCVBasicCalibration.adapt_to_resolution

```python
adapt_to_resolution(self, eye_w: int, eye_h: int) -> None
```

Adjust intrinsics so they match the actual capture eye-size.

Default policy: if the capture size is a uniform scale of the
calibration size, rescale K/image_size by that factor; otherwise
raise. ``D`` (distortion coefficients) is scale-invariant under
uniform image rescale for both pinhole-radtan and Kannala–Brandt.

Subclasses override only when they need different behaviour
(e.g. an image-size-invariant calibration that no-ops here).

#### OpenCVBasicCalibration.set_canvas_fit_mode

```python
set_canvas_fit_mode(self, mode: str) -> None
```

#### OpenCVBasicCalibration.compute_canvas_remap

```python
compute_canvas_remap(self, canvas_w: int, canvas_h: int) -> None
```

Build a combined rectify + canvas-fit remap (cv2.remap from raw
eye directly to canvas-sized rectified image, in one pass).

Generic w.r.t. concrete calibration kind: uses the K/D/R/P
produced by `_compute_rectification` and dispatches the actual
kernel via `_init_undistort_rectify_map`. On the mono path
only the left maps are built; the right buffer stays unallocated
and :meth:`canvas_remap` returns ``(canvas_l, None)``.

#### OpenCVBasicCalibration.canvas_remap

```python
canvas_remap(self, raw_l: numpy.ndarray, raw_r: Optional[numpy.ndarray]) -> tuple[numpy.ndarray, typing.Optional[numpy.ndarray]]
```

Rectify + scale-to-canvas in one cv2.remap pass per eye.

┌──────────────────────────────────────────────────────────┐
│ DANGER — ALIASED RETURN                                  │
│ The returned arrays are views into internal pre-          │
│ allocated buffers. They become INVALID on the next call  │
│ to canvas_remap (the buffers are reused). Lifetime  │
│ ends when:                                               │
│   • the next canvas_remap call starts, OR           │
│   • compute_canvas_remap reallocates buffers (resize), OR│
│   • the Calibration instance is GC'd.                    │
│ If you need to retain past any of those, use             │
│ `canvas_remap_copy` instead, or `.copy()` the views │
│ explicitly. Anything that holds the views across worker  │
│ iterations is a silent corruption bug.                   │
└──────────────────────────────────────────────────────────┘

Pipeline's worker is the only caller that uses this safely —
it consumes the views inside the same iteration (model
preprocess copies into a fresh tensor; rect_l/rect_r get
explicit `.copy()` before going into the inflight slot).

#### OpenCVBasicCalibration.canvas_remap_copy

```python
canvas_remap_copy(self, raw_l: numpy.ndarray, raw_r: Optional[numpy.ndarray]) -> tuple[numpy.ndarray, typing.Optional[numpy.ndarray]]
```

Safe variant of `canvas_remap` — returns fresh
copies that have no lifetime constraint.

Use this when you don't fully understand or can't enforce
the aliasing contract documented on `canvas_remap`.
Costs one extra allocation pair per call (~3 MB at
canvas 1280×720); negligible vs the cv2.remap itself for
correctness-critical or low-fps consumers.

#### OpenCVBasicCalibration.rectify

```python
rectify(self, left: numpy.ndarray, right: Optional[numpy.ndarray] = None) -> tuple[numpy.ndarray, typing.Optional[numpy.ndarray]]
```

Native-res rectified pair. Fresh arrays each call.

Diagnostic / API-parity utility. The pipeline worker uses
:meth:`canvas_remap` instead — it fuses rectify + canvas-fit
into one cv2.remap. On mono calibrations (or when ``right`` is
None) returns ``(rectified_left, None)``.

#### OpenCVBasicCalibration.disparity_to_depth

```python
disparity_to_depth(self, disparity_canvas_px: numpy.ndarray) -> numpy.ndarray
```

Convert canvas-pixel disparity to metric depth.

Returns a fresh array each call (FramePacket.depth is held in
the output queue; reusing a buffer would alias multiple packets).
Default: pinhole formula `Z = baseline · focal / disparity`.
Override for non-pinhole models. Raises on mono calibrations
(depth comes from a monocular depth inference instead).

#### OpenCVBasicCalibration.depth_to_point_cloud

```python
depth_to_point_cloud(self, depth_m: numpy.ndarray) -> numpy.ndarray
```

Back-project depth to a 3D point cloud (h, w, 3) in meters.

Default: pinhole unprojection using rectified P_left intrinsics.
Override for fisheye / omnidir.

#### OpenCVBasicCalibration.control_get

```python
control_get(self, name: str) -> Any
```

Return the current value of one control by name — reads its
``_control_<name>`` backing attribute. Equivalent to ``host.<name>``
(the descriptor) but takes the name as a string.

#### OpenCVBasicCalibration.controls_to_dict

```python
controls_to_dict(self) -> dict
```

Snapshot every current control value — for the UI live-state
pane, recorder sidecar, etc.

#### OpenCVBasicCalibration.controls

```python
controls(self) -> tuple[depz_camera_sdk._controls.Control, ...]
```

This host's control descriptors — the class-level
:data:`_CONTROL_SPECS` tuple (same object every call, no per-call
allocation). The single public accessor shared by every
ControlOwner: filters, inference, calibration, pipeline.

### OpenCVFisheyeCalibration

```python
class OpenCVFisheyeCalibration(*, horizontal_fov_rad: 'float | None' = None, vertical_fov_rad: 'float | None' = None, canvas_fit_mode: 'str | None' = None) -> None
```

Kannala–Brandt fisheye, lon/lat-rectified.

**Runtime controls** — read via the attribute, set via the owner's `set_*` API:

| control | type | default | range / values | unit |
|---|---|---|---|---|
| `canvas_fit_mode` | str | `fit-with-padding-middle` | `fit-with-padding-middle`, `fit-with-padding-upper-left`, `fill-with-cropping` | — |
| `horizontal_fov_rad` | float | `3.141592653589793` | 0.0001…6.283185307179586 step 0.001 | rad |
| `vertical_fov_rad` | float | `3.141592653589793` | 0.0001…6.283185307179586 step 0.001 | rad |

#### OpenCVFisheyeCalibration.compute_canvas_remap

```python
compute_canvas_remap(self, canvas_w: int, canvas_h: int) -> None
```

Build an ERP remap from raw fisheye → canvas-sized
equirectangular image, for both eyes.

#### OpenCVFisheyeCalibration.disparity_to_depth

```python
disparity_to_depth(self, disparity_canvas_px: numpy.ndarray) -> numpy.ndarray
```

Convert ERP horizontal disparity (canvas pixels) to slant
range (meters). Uses the spherical-stereo triangulation:

    r_L = b · cos(θ_R) / (cos(φ) · sin(Δθ))

where θ_L, θ_R are the left/right azimuth angles for the same
scene point, Δθ = θ_L − θ_R is the angular disparity, φ is the
elevation, and b is the baseline.

Returns slant range, NOT z-forward depth. For a flat-image
depth-like buffer use ``r * cos(θ_L) * cos(φ)`` if needed.

#### OpenCVFisheyeCalibration.depth_to_point_cloud

```python
depth_to_point_cloud(self, range_m: numpy.ndarray) -> numpy.ndarray
```

Back-project ERP slant range to 3D points in the rectified-
left frame (X right along baseline, Y down, Z forward).

#### OpenCVFisheyeCalibration.SDK_PARAMS *(property)*

Names of every runtime control. Derived from :meth:`controls`
— kept as a property (not ClassVar any more) for backwards
compatibility with callers that snapshot the namespace.

#### OpenCVFisheyeCalibration.is_loaded *(property)*

#### OpenCVFisheyeCalibration.baseline_m *(property)*

#### OpenCVFisheyeCalibration.focal_native_px *(property)*

#### OpenCVFisheyeCalibration.is_stereo *(property)*

True when the calibration was built from two cameras (left +
right, baseline + transform). False for a single-camera (mono)
calibration — no right-eye intrinsics, no baseline. Mono
pipelines route depth from the inference instead of computing
it via :meth:`disparity_to_depth`.

#### OpenCVFisheyeCalibration.canvas_layout *(property)*

Geometry of the rectified eye inside the canvas. ``None``
until :meth:`compute_canvas_remap` runs; reset to ``None`` when
rectification params change.

#### OpenCVFisheyeCalibration.P_left_canvas *(property)*

Canvas-space projection matrix of the left rectified eye —
a 3×4 ``[K_canvas | t]`` (intrinsics in canvas pixels, plus
baseline-translation column for stereo / zeros for mono).

``None`` until :meth:`compute_canvas_remap` runs. Read by
focus-aware mono inference backends (UniDepth, Metric3D) so
they can ingest the right intrinsics for the canvas the
pipeline asks them to consume.

#### OpenCVFisheyeCalibration.can_handle *(classmethod)*

```python
can_handle(cls, d: dict) -> bool
```

True when any camera in the dict advertises this class's
:attr:`POLYMORPHIC_NAME`. That's the sole identification signal —
no ``kind`` tag, no structural sniffing.

#### OpenCVFisheyeCalibration.from_dict *(classmethod)*

```python
from_dict(cls, d: dict) -> 'Calibration'
```

Build a ``Calibration`` from a cereal-shaped dict.

- Called on the abstract base (``Calibration.from_dict(d)``):
  dispatches to the registered impl whose :meth:`can_handle`
  returns True (first match wins).
- Called on a concrete subclass: directly constructs an
  instance via :meth:`_populate` + :meth:`_compute_rectification`.

#### OpenCVFisheyeCalibration.load

```python
load(self, data: dict) -> None
```

Populate from a cereal-shaped dict and recompute rectification.
Convenience wrapper for callers that already have a constructed
instance (e.g. a subclass picked up from config) and want to
mutate it in place rather than building a new one via
:meth:`from_dict`.

#### OpenCVFisheyeCalibration.to_dict

```python
to_dict(self) -> dict
```

Emit the standard cereal-shaped dict. Subclasses don't
usually need to override — only :meth:`_distortion_params_dict`.

Mono calibrations emit a single-camera entry with no ``role``
key (`role=None`) and `camera_index=0`. Stereo emits the
``role="left"`` / ``role="right"`` pair with indices 0 / 1.

#### OpenCVFisheyeCalibration.apply_update

```python
apply_update(self, patch: dict) -> None
```

Apply an online-recalibrator patch in-place.

Two forms supported:
  • Full-state — patch contains a cereal-shaped ``cameras`` list
    (anywhere via :meth:`_extract_cameras`). Re-populate
    everything and recompute rectification.
  • Partial — class-specific subset; dispatched to
    :meth:`_apply_partial`.

Pipeline drains in-flight inference + resets stateful filters
BEFORE calling this, so impls can rebuild eagerly without locks.

#### OpenCVFisheyeCalibration.adapt_to_resolution

```python
adapt_to_resolution(self, eye_w: int, eye_h: int) -> None
```

Adjust intrinsics so they match the actual capture eye-size.

Default policy: if the capture size is a uniform scale of the
calibration size, rescale K/image_size by that factor; otherwise
raise. ``D`` (distortion coefficients) is scale-invariant under
uniform image rescale for both pinhole-radtan and Kannala–Brandt.

Subclasses override only when they need different behaviour
(e.g. an image-size-invariant calibration that no-ops here).

#### OpenCVFisheyeCalibration.control_set

```python
control_set(self, key: str, value: Any) -> Any
```

Update one named runtime control and rebuild geometry. Same
side-effect-on-control_set pattern as :class:`Filter` /
:class:`DepthInferenceBase` — base impl coerces and stores, this
override layers on the geometry rebuild.

Side-effect dispatch: :meth:`_compute_rectification` for numeric
controls, canvas-state invalidation for ``canvas_fit_mode`` (caller
rebuilds the remap separately, since canvas size also varies).

Subclasses override only when they need special semantics
(sentinels, cross-control constraints) — call ``super().control_set``
for the unaffected branches.

#### OpenCVFisheyeCalibration.set_canvas_fit_mode

```python
set_canvas_fit_mode(self, mode: str) -> None
```

#### OpenCVFisheyeCalibration.canvas_remap

```python
canvas_remap(self, raw_l: numpy.ndarray, raw_r: Optional[numpy.ndarray]) -> tuple[numpy.ndarray, typing.Optional[numpy.ndarray]]
```

Rectify + scale-to-canvas in one cv2.remap pass per eye.

┌──────────────────────────────────────────────────────────┐
│ DANGER — ALIASED RETURN                                  │
│ The returned arrays are views into internal pre-          │
│ allocated buffers. They become INVALID on the next call  │
│ to canvas_remap (the buffers are reused). Lifetime  │
│ ends when:                                               │
│   • the next canvas_remap call starts, OR           │
│   • compute_canvas_remap reallocates buffers (resize), OR│
│   • the Calibration instance is GC'd.                    │
│ If you need to retain past any of those, use             │
│ `canvas_remap_copy` instead, or `.copy()` the views │
│ explicitly. Anything that holds the views across worker  │
│ iterations is a silent corruption bug.                   │
└──────────────────────────────────────────────────────────┘

Pipeline's worker is the only caller that uses this safely —
it consumes the views inside the same iteration (model
preprocess copies into a fresh tensor; rect_l/rect_r get
explicit `.copy()` before going into the inflight slot).

#### OpenCVFisheyeCalibration.canvas_remap_copy

```python
canvas_remap_copy(self, raw_l: numpy.ndarray, raw_r: Optional[numpy.ndarray]) -> tuple[numpy.ndarray, typing.Optional[numpy.ndarray]]
```

Safe variant of `canvas_remap` — returns fresh
copies that have no lifetime constraint.

Use this when you don't fully understand or can't enforce
the aliasing contract documented on `canvas_remap`.
Costs one extra allocation pair per call (~3 MB at
canvas 1280×720); negligible vs the cv2.remap itself for
correctness-critical or low-fps consumers.

#### OpenCVFisheyeCalibration.rectify

```python
rectify(self, left: numpy.ndarray, right: Optional[numpy.ndarray] = None) -> tuple[numpy.ndarray, typing.Optional[numpy.ndarray]]
```

Native-res rectified pair. Fresh arrays each call.

Diagnostic / API-parity utility. The pipeline worker uses
:meth:`canvas_remap` instead — it fuses rectify + canvas-fit
into one cv2.remap. On mono calibrations (or when ``right`` is
None) returns ``(rectified_left, None)``.

#### OpenCVFisheyeCalibration.control_get

```python
control_get(self, name: str) -> Any
```

Return the current value of one control by name — reads its
``_control_<name>`` backing attribute. Equivalent to ``host.<name>``
(the descriptor) but takes the name as a string.

#### OpenCVFisheyeCalibration.controls_to_dict

```python
controls_to_dict(self) -> dict
```

Snapshot every current control value — for the UI live-state
pane, recorder sidecar, etc.

#### OpenCVFisheyeCalibration.controls

```python
controls(self) -> tuple[depz_camera_sdk._controls.Control, ...]
```

This host's control descriptors — the class-level
:data:`_CONTROL_SPECS` tuple (same object every call, no per-call
allocation). The single public accessor shared by every
ControlOwner: filters, inference, calibration, pipeline.

### register_calibration

```python
register_calibration(cls: type) -> type
```

Decorator: register a concrete ``Calibration`` subclass.

The class must declare its own ``POLYMORPHIC_NAME`` (the cereal
lens-model identifier such as ``"LENSMODEL_OPENCV8"``). The registry
is keyed by that name — the same string that appears on
``cameras[i].model.polymorphic_name`` in the wire format and the
sole signal :meth:`Calibration.from_dict` uses to dispatch.

Idempotent re-registration of the *same class object* is a no-op
(supports notebook hot-reload where the module is re-imported and
the decorator runs again on the same class). Registering a
*different* class under an already-claimed name raises — that's
almost always a collision the operator wants to know about.

### registered_polymorphic_names

```python
registered_polymorphic_names() -> tuple[str, ...]
```

List every registered cereal ``polymorphic_name``.

## Filters

### Filter

```python
class Filter() -> None
```

One depth/disparity post-processing stage.

Lifecycle (per pipeline):
  - construction: filter holds default param values, ``enabled=False``.
  - per-frame: pipeline calls ``apply(frame)`` (no-op when disabled).
  - on seek / canvas resize / disable→enable flip: pipeline calls
    ``reset()``.
  - param change: caller goes through pipeline command queue, which
    ultimately calls ``set(name, value)`` on the filter from the
    worker thread.

Subclass contract:
  - set ``SLUG`` (canonical id, e.g. ``"median"``) — the stable
    wire identifier for RPC routing + UI introspection. Same role
    as engine slugs (``"openvino-CPU"``).
  - inherit from one of :class:`DisparityFilter` /
    :class:`DepthFilter` / :class:`PointCloudFilter` to declare
    ``OPERATES_ON``; do not set ``OPERATES_ON`` directly.
  - optional ``LABEL`` (defaults to ``SLUG`` if empty).
  - declare ``_CONTROL_SPECS`` (a tuple of :class:`Control`); its first
    entry MUST be the ``"enabled"`` boolean. Read-only properties
    and default seeding are generated from it.
  - implement ``apply()``.

``control_get``/``control_set``/``controls_to_dict`` and the default
coerce-clamp-snap logic come from
:class:`~depz_camera_sdk.._controls.ControlOwner`; subclasses override
``control_set`` only for custom snapping (discrete sets,
cross-parameter constraints).

#### Filter.apply

```python
apply(self, frame: numpy.ndarray) -> numpy.ndarray
```

Run the filter on `frame`. Return a new ndarray (callers
may retain it). Must early-return when ``self.enabled`` is
False.

#### Filter.reset

```python
reset(self) -> None
```

Clear frame-to-frame state. Default no-op (most filters
have nothing to reset). Stateful filters override.

#### Filter.control_get

```python
control_get(self, name: str) -> Any
```

Return the current value of one control by name — reads its
``_control_<name>`` backing attribute. Equivalent to ``host.<name>``
(the descriptor) but takes the name as a string.

#### Filter.control_set

```python
control_set(self, name: str, value: Any) -> Any
```

Apply one control update: look up descriptor, coerce + clamp by
``kind``/``value_type``/``min``/``max``/``step``, ``setattr``
onto ``_control_<name>``. Returns the value actually stored.

Subclasses override to add side effects (e.g. SGBM rebuilds
its matcher when controls change) — call ``super().control_set(...)``
for the coerce/store, then act on the new value.

#### Filter.controls_to_dict

```python
controls_to_dict(self) -> dict
```

Snapshot every current control value — for the UI live-state
pane, recorder sidecar, etc.

#### Filter.controls

```python
controls(self) -> tuple[depz_camera_sdk._controls.Control, ...]
```

This host's control descriptors — the class-level
:data:`_CONTROL_SPECS` tuple (same object every call, no per-call
allocation). The single public accessor shared by every
ControlOwner: filters, inference, calibration, pipeline.

### DisparityFilter

```python
class DisparityFilter() -> None
```

Runs on the raw disparity map (h1 × w1 float32 pixels) before
the calibration projects disparity → depth.

#### DisparityFilter.apply

```python
apply(self, frame: numpy.ndarray) -> numpy.ndarray
```

Run the filter on `frame`. Return a new ndarray (callers
may retain it). Must early-return when ``self.enabled`` is
False.

#### DisparityFilter.reset

```python
reset(self) -> None
```

Clear frame-to-frame state. Default no-op (most filters
have nothing to reset). Stateful filters override.

#### DisparityFilter.control_get

```python
control_get(self, name: str) -> Any
```

Return the current value of one control by name — reads its
``_control_<name>`` backing attribute. Equivalent to ``host.<name>``
(the descriptor) but takes the name as a string.

#### DisparityFilter.control_set

```python
control_set(self, name: str, value: Any) -> Any
```

Apply one control update: look up descriptor, coerce + clamp by
``kind``/``value_type``/``min``/``max``/``step``, ``setattr``
onto ``_control_<name>``. Returns the value actually stored.

Subclasses override to add side effects (e.g. SGBM rebuilds
its matcher when controls change) — call ``super().control_set(...)``
for the coerce/store, then act on the new value.

#### DisparityFilter.controls_to_dict

```python
controls_to_dict(self) -> dict
```

Snapshot every current control value — for the UI live-state
pane, recorder sidecar, etc.

#### DisparityFilter.controls

```python
controls(self) -> tuple[depz_camera_sdk._controls.Control, ...]
```

This host's control descriptors — the class-level
:data:`_CONTROL_SPECS` tuple (same object every call, no per-call
allocation). The single public accessor shared by every
ControlOwner: filters, inference, calibration, pipeline.

### DepthFilter

```python
class DepthFilter() -> None
```

Runs on the depth map (h1 × w1 float32 meters) after disparity
→ depth projection but before depth → cloud projection.

#### DepthFilter.apply

```python
apply(self, frame: numpy.ndarray) -> numpy.ndarray
```

Run the filter on `frame`. Return a new ndarray (callers
may retain it). Must early-return when ``self.enabled`` is
False.

#### DepthFilter.reset

```python
reset(self) -> None
```

Clear frame-to-frame state. Default no-op (most filters
have nothing to reset). Stateful filters override.

#### DepthFilter.control_get

```python
control_get(self, name: str) -> Any
```

Return the current value of one control by name — reads its
``_control_<name>`` backing attribute. Equivalent to ``host.<name>``
(the descriptor) but takes the name as a string.

#### DepthFilter.control_set

```python
control_set(self, name: str, value: Any) -> Any
```

Apply one control update: look up descriptor, coerce + clamp by
``kind``/``value_type``/``min``/``max``/``step``, ``setattr``
onto ``_control_<name>``. Returns the value actually stored.

Subclasses override to add side effects (e.g. SGBM rebuilds
its matcher when controls change) — call ``super().control_set(...)``
for the coerce/store, then act on the new value.

#### DepthFilter.controls_to_dict

```python
controls_to_dict(self) -> dict
```

Snapshot every current control value — for the UI live-state
pane, recorder sidecar, etc.

#### DepthFilter.controls

```python
controls(self) -> tuple[depz_camera_sdk._controls.Control, ...]
```

This host's control descriptors — the class-level
:data:`_CONTROL_SPECS` tuple (same object every call, no per-call
allocation). The single public accessor shared by every
ControlOwner: filters, inference, calibration, pipeline.

### PointCloudFilter

```python
class PointCloudFilter() -> None
```

Runs on the organized point cloud (h1 × w1 × 3 float32 meters,
in the calibration's metric frame) after depth → cloud projection.

Filters here preserve the organized h×w×3 shape so the wire
protocol stays stable; out-of-region points are zeroed rather
than removed. Downstream consumers (renderer, recorder) skip
zero points.

#### PointCloudFilter.apply

```python
apply(self, frame: numpy.ndarray) -> numpy.ndarray
```

Run the filter on `frame`. Return a new ndarray (callers
may retain it). Must early-return when ``self.enabled`` is
False.

#### PointCloudFilter.reset

```python
reset(self) -> None
```

Clear frame-to-frame state. Default no-op (most filters
have nothing to reset). Stateful filters override.

#### PointCloudFilter.control_get

```python
control_get(self, name: str) -> Any
```

Return the current value of one control by name — reads its
``_control_<name>`` backing attribute. Equivalent to ``host.<name>``
(the descriptor) but takes the name as a string.

#### PointCloudFilter.control_set

```python
control_set(self, name: str, value: Any) -> Any
```

Apply one control update: look up descriptor, coerce + clamp by
``kind``/``value_type``/``min``/``max``/``step``, ``setattr``
onto ``_control_<name>``. Returns the value actually stored.

Subclasses override to add side effects (e.g. SGBM rebuilds
its matcher when controls change) — call ``super().control_set(...)``
for the coerce/store, then act on the new value.

#### PointCloudFilter.controls_to_dict

```python
controls_to_dict(self) -> dict
```

Snapshot every current control value — for the UI live-state
pane, recorder sidecar, etc.

#### PointCloudFilter.controls

```python
controls(self) -> tuple[depz_camera_sdk._controls.Control, ...]
```

This host's control descriptors — the class-level
:data:`_CONTROL_SPECS` tuple (same object every call, no per-call
allocation). The single public accessor shared by every
ControlOwner: filters, inference, calibration, pipeline.

### SpeckleFilter

```python
class SpeckleFilter() -> None
```

OpenCV's filterSpeckles — drops tiny disconnected components.

`max_size` is the largest island (in pixels) that gets zeroed.
`max_diff` is the connectivity threshold in disparity units.

**Runtime controls** — read via the attribute, set via the owner's `set_*` API:

| control | type | default | range / values | unit |
|---|---|---|---|---|
| `enabled` | bool | `False` | — | — |
| `max_size` | int | `50` | 1…2000 step 1 | px |
| `max_diff` | float | `1.0` | 0.0…10.0 step 0.1 | disp |

#### SpeckleFilter.apply

```python
apply(self, frame: numpy.ndarray) -> numpy.ndarray
```

#### SpeckleFilter.reset

```python
reset(self) -> None
```

Clear frame-to-frame state. Default no-op (most filters
have nothing to reset). Stateful filters override.

#### SpeckleFilter.control_get

```python
control_get(self, name: str) -> Any
```

Return the current value of one control by name — reads its
``_control_<name>`` backing attribute. Equivalent to ``host.<name>``
(the descriptor) but takes the name as a string.

#### SpeckleFilter.control_set

```python
control_set(self, name: str, value: Any) -> Any
```

Apply one control update: look up descriptor, coerce + clamp by
``kind``/``value_type``/``min``/``max``/``step``, ``setattr``
onto ``_control_<name>``. Returns the value actually stored.

Subclasses override to add side effects (e.g. SGBM rebuilds
its matcher when controls change) — call ``super().control_set(...)``
for the coerce/store, then act on the new value.

#### SpeckleFilter.controls_to_dict

```python
controls_to_dict(self) -> dict
```

Snapshot every current control value — for the UI live-state
pane, recorder sidecar, etc.

#### SpeckleFilter.controls

```python
controls(self) -> tuple[depz_camera_sdk._controls.Control, ...]
```

This host's control descriptors — the class-level
:data:`_CONTROL_SPECS` tuple (same object every call, no per-call
allocation). The single public accessor shared by every
ControlOwner: filters, inference, calibration, pipeline.

### ZClipFilter

```python
class ZClipFilter() -> None
```

Zero out depth pixels outside ``[min, max]`` meters.

Destructive — downstream filters and the point cloud see the
clipped depth. For a non-destructive view-only equivalent, the
web viewer's render config exposes ``min_depth`` / ``max_depth``
that drive colormap normalization + cloud near/far culling at
render time without touching the underlying data.

**Runtime controls** — read via the attribute, set via the owner's `set_*` API:

| control | type | default | range / values | unit |
|---|---|---|---|---|
| `enabled` | bool | `False` | — | — |
| `min` | float | `0.3` | 0.05…20.0 step 0.05 | m |
| `max` | float | `5.0` | 0.1…20.0 step 0.1 | m |

#### ZClipFilter.apply

```python
apply(self, frame: numpy.ndarray) -> numpy.ndarray
```

#### ZClipFilter.control_set

```python
control_set(self, name: str, value)
```

#### ZClipFilter.reset

```python
reset(self) -> None
```

Clear frame-to-frame state. Default no-op (most filters
have nothing to reset). Stateful filters override.

#### ZClipFilter.control_get

```python
control_get(self, name: str) -> Any
```

Return the current value of one control by name — reads its
``_control_<name>`` backing attribute. Equivalent to ``host.<name>``
(the descriptor) but takes the name as a string.

#### ZClipFilter.controls_to_dict

```python
controls_to_dict(self) -> dict
```

Snapshot every current control value — for the UI live-state
pane, recorder sidecar, etc.

#### ZClipFilter.controls

```python
controls(self) -> tuple[depz_camera_sdk._controls.Control, ...]
```

This host's control descriptors — the class-level
:data:`_CONTROL_SPECS` tuple (same object every call, no per-call
allocation). The single public accessor shared by every
ControlOwner: filters, inference, calibration, pipeline.

### EdgeFilter

```python
class EdgeFilter() -> None
```

Zero out pixels at depth discontinuities (edge skirts / flyers).

Keeps per-shape scratch buffers between calls so the per-frame
work is allocation-free in steady state. Buffers re-allocate on
canvas resize; :meth:`reset` clears them.

**Runtime controls** — read via the attribute, set via the owner's `set_*` API:

| control | type | default | range / values | unit |
|---|---|---|---|---|
| `enabled` | bool | `False` | — | — |
| `rel_thresh` | float | `0.05` | 0.001…2.0 step 0.005 | — |

#### EdgeFilter.reset

```python
reset(self) -> None
```

#### EdgeFilter.apply

```python
apply(self, frame: numpy.ndarray) -> numpy.ndarray
```

#### EdgeFilter.control_get

```python
control_get(self, name: str) -> Any
```

Return the current value of one control by name — reads its
``_control_<name>`` backing attribute. Equivalent to ``host.<name>``
(the descriptor) but takes the name as a string.

#### EdgeFilter.control_set

```python
control_set(self, name: str, value: Any) -> Any
```

Apply one control update: look up descriptor, coerce + clamp by
``kind``/``value_type``/``min``/``max``/``step``, ``setattr``
onto ``_control_<name>``. Returns the value actually stored.

Subclasses override to add side effects (e.g. SGBM rebuilds
its matcher when controls change) — call ``super().control_set(...)``
for the coerce/store, then act on the new value.

#### EdgeFilter.controls_to_dict

```python
controls_to_dict(self) -> dict
```

Snapshot every current control value — for the UI live-state
pane, recorder sidecar, etc.

#### EdgeFilter.controls

```python
controls(self) -> tuple[depz_camera_sdk._controls.Control, ...]
```

This host's control descriptors — the class-level
:data:`_CONTROL_SPECS` tuple (same object every call, no per-call
allocation). The single public accessor shared by every
ControlOwner: filters, inference, calibration, pipeline.

### TemporalEmaFilter

```python
class TemporalEmaFilter() -> None
```

Per-pixel EMA blending of depth with the previous frame.

`alpha` weights the current frame: ``out = α·cur + (1-α)·prev``.
α=1 is "off" (current passes through). Both pixels must be valid
(>0) to blend; otherwise the current pixel passes through.

**Runtime controls** — read via the attribute, set via the owner's `set_*` API:

| control | type | default | range / values | unit |
|---|---|---|---|---|
| `enabled` | bool | `False` | — | — |
| `alpha` | float | `0.3` | 0.05…1.0 step 0.05 | — |

#### TemporalEmaFilter.reset

```python
reset(self) -> None
```

#### TemporalEmaFilter.control_set

```python
control_set(self, name: str, value)
```

#### TemporalEmaFilter.apply

```python
apply(self, frame: numpy.ndarray) -> numpy.ndarray
```

#### TemporalEmaFilter.control_get

```python
control_get(self, name: str) -> Any
```

Return the current value of one control by name — reads its
``_control_<name>`` backing attribute. Equivalent to ``host.<name>``
(the descriptor) but takes the name as a string.

#### TemporalEmaFilter.controls_to_dict

```python
controls_to_dict(self) -> dict
```

Snapshot every current control value — for the UI live-state
pane, recorder sidecar, etc.

#### TemporalEmaFilter.controls

```python
controls(self) -> tuple[depz_camera_sdk._controls.Control, ...]
```

This host's control descriptors — the class-level
:data:`_CONTROL_SPECS` tuple (same object every call, no per-call
allocation). The single public accessor shared by every
ControlOwner: filters, inference, calibration, pipeline.

### BoundingBoxFilter

```python
class BoundingBoxFilter() -> None
```

Drop points outside an axis-aligned 3D box (in the calibration's
metric frame, typically meters in the camera-rig frame).

Strict superset of :class:`ZClipFilter` (which only handles the Z
axis on depth). Useful when the consumer cares about a specific
region of space — robotics workspace, AR table, etc.

Out-of-box points are zeroed (XYZ all set to 0); the organized
h×w×3 shape stays intact so the wire protocol is unchanged.

**Runtime controls** — read via the attribute, set via the owner's `set_*` API:

| control | type | default | range / values | unit |
|---|---|---|---|---|
| `enabled` | bool | `False` | — | — |
| `x_min` | float | `-5.0` | -20.0…20.0 step 0.05 | m |
| `x_max` | float | `5.0` | -20.0…20.0 step 0.05 | m |
| `y_min` | float | `-5.0` | -20.0…20.0 step 0.05 | m |
| `y_max` | float | `5.0` | -20.0…20.0 step 0.05 | m |
| `z_min` | float | `0.1` | 0.05…20.0 step 0.05 | m |
| `z_max` | float | `10.0` | 0.1…20.0 step 0.1 | m |

#### BoundingBoxFilter.apply

```python
apply(self, frame: numpy.ndarray) -> numpy.ndarray
```

#### BoundingBoxFilter.control_set

```python
control_set(self, name: str, value)
```

#### BoundingBoxFilter.reset

```python
reset(self) -> None
```

Clear frame-to-frame state. Default no-op (most filters
have nothing to reset). Stateful filters override.

#### BoundingBoxFilter.control_get

```python
control_get(self, name: str) -> Any
```

Return the current value of one control by name — reads its
``_control_<name>`` backing attribute. Equivalent to ``host.<name>``
(the descriptor) but takes the name as a string.

#### BoundingBoxFilter.controls_to_dict

```python
controls_to_dict(self) -> dict
```

Snapshot every current control value — for the UI live-state
pane, recorder sidecar, etc.

#### BoundingBoxFilter.controls

```python
controls(self) -> tuple[depz_camera_sdk._controls.Control, ...]
```

This host's control descriptors — the class-level
:data:`_CONTROL_SPECS` tuple (same object every call, no per-call
allocation). The single public accessor shared by every
ControlOwner: filters, inference, calibration, pipeline.

### StatisticalOutlierFilter

```python
class StatisticalOutlierFilter() -> None
```

Drop points whose local 3D neighborhood is much sparser than
the cloud's average — the classic "flying pixel" remover.

For the organized cloud, "k nearest neighbors" reduces to the 8
grid neighbors at (±1, ±1). For each non-zero pixel, we compute
the mean Euclidean distance to its valid (non-zero) grid
neighbors, then drop pixels whose mean distance exceeds the
global ``μ + std_ratio · σ`` of those means.

Catches points that survived :class:`EdgeFilter` (which only
looks at depth gradients in 2D) but are isolated in 3D — typical
flyers at object boundaries after stereo inference noise.

Controls:
  - `std_ratio` — pixels with mean distance > μ + std_ratio·σ are
    outliers. Default 1.0 (aggressive). Higher = more permissive.
  - `min_neighbors` — pixels with fewer valid grid-neighbors are
    treated as outliers regardless of distance (corners /
    isolated patches).

**Runtime controls** — read via the attribute, set via the owner's `set_*` API:

| control | type | default | range / values | unit |
|---|---|---|---|---|
| `enabled` | bool | `False` | — | — |
| `std_ratio` | float | `1.0` | 0.1…5.0 step 0.05 | — |
| `min_neighbors` | int | `3` | 1…8 step 1 | — |

#### StatisticalOutlierFilter.apply

```python
apply(self, frame: numpy.ndarray) -> numpy.ndarray
```

#### StatisticalOutlierFilter.reset

```python
reset(self) -> None
```

Clear frame-to-frame state. Default no-op (most filters
have nothing to reset). Stateful filters override.

#### StatisticalOutlierFilter.control_get

```python
control_get(self, name: str) -> Any
```

Return the current value of one control by name — reads its
``_control_<name>`` backing attribute. Equivalent to ``host.<name>``
(the descriptor) but takes the name as a string.

#### StatisticalOutlierFilter.control_set

```python
control_set(self, name: str, value: Any) -> Any
```

Apply one control update: look up descriptor, coerce + clamp by
``kind``/``value_type``/``min``/``max``/``step``, ``setattr``
onto ``_control_<name>``. Returns the value actually stored.

Subclasses override to add side effects (e.g. SGBM rebuilds
its matcher when controls change) — call ``super().control_set(...)``
for the coerce/store, then act on the new value.

#### StatisticalOutlierFilter.controls_to_dict

```python
controls_to_dict(self) -> dict
```

Snapshot every current control value — for the UI live-state
pane, recorder sidecar, etc.

#### StatisticalOutlierFilter.controls

```python
controls(self) -> tuple[depz_camera_sdk._controls.Control, ...]
```

This host's control descriptors — the class-level
:data:`_CONTROL_SPECS` tuple (same object every call, no per-call
allocation). The single public accessor shared by every
ControlOwner: filters, inference, calibration, pipeline.

### default_filter_chain

```python
default_filter_chain() -> list[depz_camera_sdk.filters.base.Filter]
```

The pipeline's stock filter chain. Construction order matches
application order — ``OPERATES_ON`` is the only thing the worker
looks at when partitioning into the disparity / depth / point-cloud
domains.

## Cameras

### fetch_camera_info_json

```python
fetch_camera_info_json(camera_serial: str, timeout: float = 10.0) -> dict
```

Query the device-info endpoint by serial. Returns the payload
as a dict in SDK shape (``"base_calibration"`` already at the top
level — server emits this directly).

Raises if the network call fails or the response is malformed
JSON. Caller decides what to do on missing fields.

### DepzCameraInfo

```python
class DepzCameraInfo(os_id: str, serial: str, sku: str, model_name: str, spec: depz_camera_sdk.cameras.camera_spec.CameraSpec, base_calibration: depz_camera_sdk.calibration.base.Calibration) -> None
```

One attached DEPZ camera, fully resolved.

A :class:`DepzCameraInfo` is the single source of truth for a
camera's identity, capability, and calibration during a session.
Construct via :meth:`from_dict` (or via :func:`list_connected_depz_cameras`
/ :func:`get_connected_depz_camera_by_serial`); never mutate.

#### DepzCameraInfo.from_dict *(classmethod)*

```python
from_dict(cls, payload: dict, *, os_id: str) -> Optional[ForwardRef('DepzCameraInfo')]
```

Build from a SDK-shape camera-info JSON payload + the OS-side
identifier (from :meth:`PlatformBackend.list_cameras`).

The calibration block is read from ``payload["base_calibration"]``
(the inner cereal shape — same one :meth:`Calibration.to_dict`
emits). Server device-info payloads are normalized to this
shape inside :func:`fetch_camera_info_json` before reaching here.

Returns ``None`` if any required piece is missing or unrecognized:
``serial_number`` / ``sku`` not in payload, SKU not in the device
registry, or calibration block can't deserialize.

#### DepzCameraInfo.to_dict

```python
to_dict(self) -> dict
```

Emit the SDK-shape camera-info dict — the canonical wire
form a recorder embeds in MKV. Round-trips through
:meth:`from_dict`.

Server-only fields the SDK doesn't model (``calibrated_at``,
``parameters.s1..s4`` / ``tauX`` / ``tauY``, etc.) are NOT
preserved — re-fetch from the device-info endpoint if you
need them for diagnostics.

### list_connected_depz_cameras

```python
list_connected_depz_cameras() -> list[depz_camera_sdk.cameras.camera_info.DepzCameraInfo]
```

Return every connected video device that the device-info endpoint
recognizes as a registered DEPZ camera class.

Iterates :meth:`PlatformBackend.list_cameras`, fetches
device-info for each serial, and keeps only those entries where
:meth:`DepzCameraInfo.from_dict` succeeds. Devices that fail are
logged and dropped — use :func:`list_connected_depz_cameras_detailed`
when the caller wants the failures as data (UI source lists).

### get_first_connected_depz_camera

```python
get_first_connected_depz_camera() -> depz_camera_sdk.cameras.camera_info.DepzCameraInfo
```

Return the first connected DEPZ camera the device-info endpoint
recognizes. Convenience over ``list_connected_depz_cameras()[0]``
for the single-camera quick-start case.

Raises :class:`RuntimeError` when nothing is connected (or no
connected device has a serial registered in the catalog) — clearer
than the ``IndexError`` you'd get from indexing an empty list.

### get_connected_depz_camera_by_serial

```python
get_connected_depz_camera_by_serial(serial: str) -> Optional[depz_camera_sdk.cameras.camera_info.DepzCameraInfo]
```

Like :func:`list_connected_depz_cameras` but bounded to one serial — runs a
single device-info fetch instead of N.

Used when the caller already knows which camera they want
(``DepthPipeline(source="<serial>")``). Returns ``None`` if no
connected device has that serial, the serial isn't in the server
catalog, or the SKU isn't a registered :class:`CameraSpec`.

### CaptureMode

```python
class CaptureMode(width: int, height: int, fps: float, fourcc: str, stereo: bool, is_default: bool = False, is_supported: bool = True) -> None
```

One capture configuration for a camera (a row in
:attr:`CameraSpec.modes`).

``width × height`` is the **raw frame** the driver delivers. For
mono modes that's the single eye; for stereo modes it's the SBS
composite (per-eye is ``width // 2 × height``). Use the
:attr:`eye_w` / :attr:`eye_h` properties when you want per-eye
dimensions regardless of family — that's also what the slug
encodes (so mono and stereo slugs read uniformly as the eye size
every user reasons about).

fps is float because V4L2 advertises fractional rates (7.5, 12.5,
...); integer values pass through unchanged.

Per-spec, exactly one mode should have ``is_default=True``. UI uses
this to preselect the dropdown entry on first load.

``is_supported`` reflects whether the SDK pipeline can run on this
geometry. The current rule: per-eye must be 16:9. Modes with other
aspects (e.g. firmware-cropped variants) are listed for
completeness but rendered greyed-out + unselectable in the UI.

#### CaptureMode.mono *(property)*

Convenience inverse of :attr:`stereo`. Most code branches on
the mono case (the special path); the field stays ``stereo``
because that's how spec authors think about the common one.

#### CaptureMode.eye_w *(property)*

Per-eye width — :attr:`width` for mono modes (the whole
frame is one eye), ``width // 2`` for stereo SBS.

#### CaptureMode.eye_h *(property)*

Per-eye height — equals :attr:`height` (SBS doesn't split
vertically). Provided as a sibling of :attr:`eye_w` so callers
can write ``mode.eye_w, mode.eye_h`` symmetrically.

#### CaptureMode.slug *(property)*

Compact wire identifier: ``"{eye_w}x{eye_h}@{fps}/{FOURCC}"``,
with ``/stereo`` appended when :attr:`stereo` is ``True``.

**Per-eye semantics**: the W×H in the slug is the dimension a
user actually sees in the UI label and reasons about (the
rectified canvas / per-eye crop), not the raw SBS frame from
the driver. Mono and stereo slugs read uniformly:

    "1280x720@30/MJPG/stereo"   # stereo, eye 1280×720 (raw SBS 2560×720)
    "640x360@7.5/YUYV/stereo"   # stereo, eye 640×360  (raw SBS 1280×360)
    "1280x720@10/MJPG"          # mono,   eye 1280×720 (raw 1280×720)

The ``/stereo`` suffix disambiguates families with the same
eye size — `1280x720@10/MJPG` (mono single-eye) is a different
mode from `1280x720@10/MJPG/stereo` (SBS where each eye is
1280×720, raw frame 2560×720). Stable across SDK versions —
:meth:`CameraSpec.get_mode` parses it back, so UI dropdown
values double as round-trip keys without a separate id table.

### CameraControl

```python
class CameraControl(kind: str, value_type: str, default: Any = None, label: str = '', min: float | None = None, max: float | None = None, step: float | None = None, enum_values: tuple[tuple[int | float | str, str], ...] = (), unit: str = '', name: str = '', current_value: Optional[float] = None, auto_companion: Optional[str] = None, updatable_by_auto: bool = True, writable: bool = True) -> None
```

Camera-control descriptor — :class:`~depz_camera_sdk._controls.Control`
extended with V4L2-cascade and runtime-snapshot fields.

Used both as a spec literal (in :attr:`CameraSpec.controls`) and as
a runtime snapshot (returned from
:meth:`DepthPipeline.list_camera_controls`). The spec is the single
source of truth for ``min`` / ``max`` / ``step`` / ``default`` /
``label`` / ``unit`` / ``enum_values`` — there is no driver
fallback. Runtime snapshots only override :attr:`current_value`.

Platform-specific control IDs (V4L2 CIDs, DirectShow keys, ...)
are *not* on this dataclass — each platform backend keeps its own
canonical-name → id map.

#### CameraControl.to_dict

```python
to_dict(self) -> dict
```

### CameraSpec

```python
class CameraSpec(modes: tuple[depz_camera_sdk.cameras.camera_spec.CaptureMode, ...], controls: tuple[depz_camera_sdk.cameras.camera_spec.CameraControl, ...], windows_controls: tuple[depz_camera_sdk.cameras.camera_spec.CameraControl, ...] = (), skus: tuple[str, ...] = ()) -> None
```

Curated description of one camera firmware-class. Identified
by its :attr:`skus` — the camera-info endpoint maps a unit's
serial to a SKU, and :func:`spec_by_sku` walks that SKU back to
the matching spec.

#### CameraSpec.platform_controls *(property)*

Camera controls for the *current* host OS. Windows reports
different units/ranges for some controls than V4L2 — see
:attr:`windows_controls`. Returns :attr:`controls` on Linux, and
on Windows when no :attr:`windows_controls` table is set.

This is the accessor every consumer should use; :attr:`controls`
and :attr:`windows_controls` are the raw per-platform tables.

#### CameraSpec.default_mode *(property)*

The single :class:`CaptureMode` flagged ``is_default=True``
— what the UI preselects on first load and what
``CameraSource(cam)`` falls back to if no explicit mode is
passed. Spec invariant: exactly one mode must be marked
default; raises :class:`RuntimeError` otherwise so a malformed
spec fails loudly at use site instead of silently picking the
first match.

#### CameraSpec.get_mode

```python
get_mode(self, slug: str) -> depz_camera_sdk.cameras.camera_spec.CaptureMode
```

Look up one :class:`CaptureMode` by its :attr:`CaptureMode.slug`.

Slug format is ``"{W}x{H}@{fps}/{FOURCC}"`` with optional
``/mono`` suffix. Use this when round-tripping a mode through
the wire (UI dropdown values, RPC params, MKV sidecar) — saves
re-introspecting the spec by tuple-matching W/H/fps/fourcc/mono.

Raises :class:`KeyError` if no mode matches; the message lists
every available slug for the spec so a typo is easy to spot.

### DEPZ_VARIUS

`CameraSpec` constant.

Curated description of one camera firmware-class. Identified
by its :attr:`skus` — the camera-info endpoint maps a unit's
serial to a SKU, and :func:`spec_by_sku` walks that SKU back to
the matching spec.

### DEPZ_MINIMA

`CameraSpec` constant.

Curated description of one camera firmware-class. Identified
by its :attr:`skus` — the camera-info endpoint maps a unit's
serial to a SKU, and :func:`spec_by_sku` walks that SKU back to
the matching spec.

### DEPZ_CLARITY

`CameraSpec` constant.

Curated description of one camera firmware-class. Identified
by its :attr:`skus` — the camera-info endpoint maps a unit's
serial to a SKU, and :func:`spec_by_sku` walks that SKU back to
the matching spec.

### REGISTRY

`tuple` constant.

Built-in immutable sequence.

If no argument is given, the constructor returns an empty tuple.
If iterable is specified the tuple is initialized from iterable's items.

If the argument is a tuple, the return value is the same object.

### spec_by_sku

```python
spec_by_sku(sku: str, registry: tuple[depz_camera_sdk.cameras.camera_spec.CameraSpec, ...] = (CameraSpec(modes=(CaptureMode(width=1280, height=720, fps=30, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=25, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=20, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=15, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=10, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=7.5, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=5, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=480, fps=30, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=25, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=20, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=15, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=10, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=7.5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=360, fps=30, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=25, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=20, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=15, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=10, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=7.5, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=5, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=712, fps=30, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=712, fps=25, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=712, fps=20, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=712, fps=15, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=712, fps=10, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=712, fps=7.5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=712, fps=5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=30, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=25, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=20, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=15, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=10, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=7.5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=30, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=25, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=20, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=15, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=10, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=7.5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=2560, height=720, fps=30, fourcc='MJPG', stereo=True, is_default=False, is_supported=True), CaptureMode(width=2560, height=720, fps=25, fourcc='MJPG', stereo=True, is_default=False, is_supported=True), CaptureMode(width=2560, height=720, fps=20, fourcc='MJPG', stereo=True, is_default=False, is_supported=True), CaptureMode(width=2560, height=720, fps=15, fourcc='MJPG', stereo=True, is_default=False, is_supported=True), CaptureMode(width=2560, height=720, fps=10, fourcc='MJPG', stereo=True, is_default=False, is_supported=True), CaptureMode(width=2560, height=720, fps=7.5, fourcc='MJPG', stereo=True, is_default=False, is_supported=True), CaptureMode(width=2560, height=720, fps=5, fourcc='MJPG', stereo=True, is_default=False, is_supported=True), CaptureMode(width=1280, height=480, fps=30, fourcc='MJPG', stereo=True, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=25, fourcc='MJPG', stereo=True, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=20, fourcc='MJPG', stereo=True, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=15, fourcc='MJPG', stereo=True, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=10, fourcc='MJPG', stereo=True, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=7.5, fourcc='MJPG', stereo=True, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=5, fourcc='MJPG', stereo=True, is_default=False, is_supported=False), CaptureMode(width=1280, height=360, fps=30, fourcc='MJPG', stereo=True, is_default=True, is_supported=True), CaptureMode(width=1280, height=360, fps=25, fourcc='MJPG', stereo=True, is_default=False, is_supported=True), CaptureMode(width=1280, height=360, fps=20, fourcc='MJPG', stereo=True, is_default=False, is_supported=True), CaptureMode(width=1280, height=360, fps=15, fourcc='MJPG', stereo=True, is_default=False, is_supported=True), CaptureMode(width=1280, height=360, fps=10, fourcc='MJPG', stereo=True, is_default=False, is_supported=True), CaptureMode(width=1280, height=360, fps=7.5, fourcc='MJPG', stereo=True, is_default=False, is_supported=True), CaptureMode(width=1280, height=360, fps=5, fourcc='MJPG', stereo=True, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=10, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=7.5, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=5, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=480, fps=30, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=25, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=20, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=15, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=10, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=7.5, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=5, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=360, fps=30, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=25, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=20, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=15, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=10, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=7.5, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=5, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=712, fps=10, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=712, fps=7.5, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=712, fps=5, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=30, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=25, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=20, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=15, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=10, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=7.5, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=5, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=30, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=25, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=20, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=15, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=10, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=7.5, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=5, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=2560, height=720, fps=5, fourcc='YUYV', stereo=True, is_default=False, is_supported=True), CaptureMode(width=1280, height=480, fps=15, fourcc='YUYV', stereo=True, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=10, fourcc='YUYV', stereo=True, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=7.5, fourcc='YUYV', stereo=True, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=5, fourcc='YUYV', stereo=True, is_default=False, is_supported=False), CaptureMode(width=1280, height=360, fps=25, fourcc='YUYV', stereo=True, is_default=False, is_supported=True), CaptureMode(width=1280, height=360, fps=20, fourcc='YUYV', stereo=True, is_default=False, is_supported=True), CaptureMode(width=1280, height=360, fps=15, fourcc='YUYV', stereo=True, is_default=False, is_supported=True), CaptureMode(width=1280, height=360, fps=10, fourcc='YUYV', stereo=True, is_default=False, is_supported=True), CaptureMode(width=1280, height=360, fps=7.5, fourcc='YUYV', stereo=True, is_default=False, is_supported=True), CaptureMode(width=1280, height=360, fps=5, fourcc='YUYV', stereo=True, is_default=False, is_supported=True)), controls=(CameraControl(kind='enum', value_type='int', default=3, label='Auto exposure', min=0, max=3, step=1, enum_values=((1, 'Manual'), (3, 'Aperture Priority')), unit='', name='auto_exposure', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=312, label='Exposure', min=1, max=80000, step=1, enum_values=(), unit='100us', name='exposure', current_value=None, auto_companion='auto_exposure', updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=32, label='Gain', min=0, max=130, step=1, enum_values=(), unit='', name='gain', current_value=None, auto_companion='auto_exposure', updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=0, label='Brightness', min=-64, max=64, step=1, enum_values=(), unit='', name='brightness', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=34, label='Contrast', min=0, max=64, step=1, enum_values=(), unit='', name='contrast', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=62, label='Saturation', min=0, max=78, step=1, enum_values=(), unit='', name='saturation', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=0, label='Hue', min=-2000, max=2000, step=1, enum_values=(), unit='', name='hue', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=150, label='Gamma', min=100, max=300, step=1, enum_values=(), unit='', name='gamma', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=4, label='Sharpness', min=1, max=7, step=1, enum_values=(), unit='', name='sharpness', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='boolean', value_type='bool', default=1, label='Auto white balance', min=0, max=1, step=1, enum_values=(), unit='', name='white_balance_auto', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=4600, label='White balance (K)', min=2800, max=6500, step=1, enum_values=(), unit='K', name='white_balance', current_value=None, auto_companion='white_balance_auto', updatable_by_auto=False, writable=True), CameraControl(kind='enum', value_type='int', default=1, label='Power-line frequency', min=0, max=2, step=1, enum_values=((0, 'Disabled'), (1, '50 Hz'), (2, '60 Hz')), unit='', name='power_line_frequency', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=0, label='Backlight compensation', min=0, max=1, step=1, enum_values=(), unit='', name='backlight_compensation', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True)), windows_controls=(CameraControl(kind='enum', value_type='int', default=3, label='Auto exposure', min=0, max=3, step=1, enum_values=((1, 'Manual'), (3, 'Aperture Priority')), unit='', name='auto_exposure', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=-5, label='Exposure', min=-13, max=3, step=1, enum_values=(), unit='log2(s)', name='exposure', current_value=None, auto_companion='auto_exposure', updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=32, label='Gain', min=0, max=130, step=1, enum_values=(), unit='', name='gain', current_value=None, auto_companion='auto_exposure', updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=0, label='Brightness', min=-64, max=64, step=1, enum_values=(), unit='', name='brightness', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=34, label='Contrast', min=0, max=64, step=1, enum_values=(), unit='', name='contrast', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=62, label='Saturation', min=0, max=78, step=1, enum_values=(), unit='', name='saturation', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=0, label='Hue', min=-2000, max=2000, step=1, enum_values=(), unit='', name='hue', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=150, label='Gamma', min=100, max=300, step=1, enum_values=(), unit='', name='gamma', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=4, label='Sharpness', min=1, max=7, step=1, enum_values=(), unit='', name='sharpness', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='boolean', value_type='bool', default=1, label='Auto white balance', min=0, max=1, step=1, enum_values=(), unit='', name='white_balance_auto', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=4600, label='White balance (K)', min=2800, max=6500, step=1, enum_values=(), unit='K', name='white_balance', current_value=None, auto_companion='white_balance_auto', updatable_by_auto=False, writable=True), CameraControl(kind='number', value_type='int', default=0, label='Backlight compensation', min=0, max=1, step=1, enum_values=(), unit='', name='backlight_compensation', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True)), skus=('STEREO-CAMERA-VARIUS-ULTRA-SHORT-BASELINE-NARROW-LENSES-V21', 'STEREO-CAMERA-VARIUS-ULTRA-SHORT-BASELINE-REGULAR-LENSES-V21', 'STEREO-CAMERA-VARIUS-ULTRA-SHORT-BASELINE-WIDE-LENSES-V21', 'STEREO-CAMERA-VARIUS-ULTRA-SHORT-BASELINE-ULTRA-WIDE-LENSES-V21', 'STEREO-CAMERA-VARIUS-ULTRA-SHORT-BASELINE-FISHEYE-LENSES-V21', 'STEREO-CAMERA-VARIUS-ULTRA-SHORT-BASELINE-ULTRA-FISHEYE-LENSES-V21', 'STEREO-CAMERA-VARIUS-ULTRA-SHORT-BASELINE-PANOPTIC-LENSES-V21', 'STEREO-CAMERA-VARIUS-SHORT-BASELINE-NARROW-LENSES-V21', 'STEREO-CAMERA-VARIUS-SHORT-BASELINE-REGULAR-LENSES-V21', 'STEREO-CAMERA-VARIUS-SHORT-BASELINE-WIDE-LENSES-V21', 'STEREO-CAMERA-VARIUS-SHORT-BASELINE-ULTRA-WIDE-LENSES-V21', 'STEREO-CAMERA-VARIUS-SHORT-BASELINE-FISHEYE-LENSES-V21', 'STEREO-CAMERA-VARIUS-SHORT-BASELINE-ULTRA-FISHEYE-LENSES-V21', 'STEREO-CAMERA-VARIUS-SHORT-BASELINE-PANOPTIC-LENSES-V21', 'STEREO-CAMERA-VARIUS-REGULAR-BASELINE-NARROW-LENSES-V21', 'STEREO-CAMERA-VARIUS-REGULAR-BASELINE-REGULAR-LENSES-V21', 'STEREO-CAMERA-VARIUS-REGULAR-BASELINE-WIDE-LENSES-V21', 'STEREO-CAMERA-VARIUS-REGULAR-BASELINE-ULTRA-WIDE-LENSES-V21', 'STEREO-CAMERA-VARIUS-REGULAR-BASELINE-FISHEYE-LENSES-V21', 'STEREO-CAMERA-VARIUS-REGULAR-BASELINE-ULTRA-FISHEYE-LENSES-V21', 'STEREO-CAMERA-VARIUS-REGULAR-BASELINE-PANOPTIC-LENSES-V21', 'STEREO-CAMERA-VARIUS-LONG-BASELINE-NARROW-LENSES-V21', 'STEREO-CAMERA-VARIUS-LONG-BASELINE-REGULAR-LENSES-V21', 'STEREO-CAMERA-VARIUS-LONG-BASELINE-WIDE-LENSES-V21', 'STEREO-CAMERA-VARIUS-LONG-BASELINE-ULTRA-WIDE-LENSES-V21', 'STEREO-CAMERA-VARIUS-LONG-BASELINE-FISHEYE-LENSES-V21', 'STEREO-CAMERA-VARIUS-LONG-BASELINE-ULTRA-FISHEYE-LENSES-V21', 'STEREO-CAMERA-VARIUS-LONG-BASELINE-PANOPTIC-LENSES-V21', 'STEREO-CAMERA-MODULE-VARIUS-ULTRA-SHORT-BASELINE-NARROW-LENSES-V21', 'STEREO-CAMERA-MODULE-VARIUS-ULTRA-SHORT-BASELINE-REGULAR-LENSES-V21', 'STEREO-CAMERA-MODULE-VARIUS-ULTRA-SHORT-BASELINE-WIDE-LENSES-V21', 'STEREO-CAMERA-MODULE-VARIUS-ULTRA-SHORT-BASELINE-ULTRA-WIDE-LENSES-V21', 'STEREO-CAMERA-MODULE-VARIUS-ULTRA-SHORT-BASELINE-FISHEYE-LENSES-V21', 'STEREO-CAMERA-MODULE-VARIUS-ULTRA-SHORT-BASELINE-ULTRA-FISHEYE-LENSES-V21', 'STEREO-CAMERA-MODULE-VARIUS-ULTRA-SHORT-BASELINE-PANOPTIC-LENSES-V21', 'STEREO-CAMERA-MODULE-VARIUS-SHORT-BASELINE-NARROW-LENSES-V21', 'STEREO-CAMERA-MODULE-VARIUS-SHORT-BASELINE-REGULAR-LENSES-V21', 'STEREO-CAMERA-MODULE-VARIUS-SHORT-BASELINE-WIDE-LENSES-V21', 'STEREO-CAMERA-MODULE-VARIUS-SHORT-BASELINE-ULTRA-WIDE-LENSES-V21', 'STEREO-CAMERA-MODULE-VARIUS-SHORT-BASELINE-FISHEYE-LENSES-V21', 'STEREO-CAMERA-MODULE-VARIUS-SHORT-BASELINE-ULTRA-FISHEYE-LENSES-V21', 'STEREO-CAMERA-MODULE-VARIUS-SHORT-BASELINE-PANOPTIC-LENSES-V21', 'STEREO-CAMERA-MODULE-VARIUS-REGULAR-BASELINE-NARROW-LENSES-V21', 'STEREO-CAMERA-MODULE-VARIUS-REGULAR-BASELINE-REGULAR-LENSES-V21', 'STEREO-CAMERA-MODULE-VARIUS-REGULAR-BASELINE-WIDE-LENSES-V21', 'STEREO-CAMERA-MODULE-VARIUS-REGULAR-BASELINE-ULTRA-WIDE-LENSES-V21', 'STEREO-CAMERA-MODULE-VARIUS-REGULAR-BASELINE-FISHEYE-LENSES-V21', 'STEREO-CAMERA-MODULE-VARIUS-REGULAR-BASELINE-ULTRA-FISHEYE-LENSES-V21', 'STEREO-CAMERA-MODULE-VARIUS-REGULAR-BASELINE-PANOPTIC-LENSES-V21', 'STEREO-CAMERA-MODULE-VARIUS-LONG-BASELINE-NARROW-LENSES-V21', 'STEREO-CAMERA-MODULE-VARIUS-LONG-BASELINE-REGULAR-LENSES-V21', 'STEREO-CAMERA-MODULE-VARIUS-LONG-BASELINE-WIDE-LENSES-V21', 'STEREO-CAMERA-MODULE-VARIUS-LONG-BASELINE-ULTRA-WIDE-LENSES-V21', 'STEREO-CAMERA-MODULE-VARIUS-LONG-BASELINE-FISHEYE-LENSES-V21', 'STEREO-CAMERA-MODULE-VARIUS-LONG-BASELINE-ULTRA-FISHEYE-LENSES-V21', 'STEREO-CAMERA-MODULE-VARIUS-LONG-BASELINE-PANOPTIC-LENSES-V21')), CameraSpec(modes=(CaptureMode(width=1280, height=720, fps=30, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=25, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=20, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=15, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=10, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=7.5, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=5, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=480, fps=30, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=25, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=20, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=15, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=10, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=7.5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=360, fps=30, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=25, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=20, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=15, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=10, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=7.5, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=5, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=712, fps=30, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=712, fps=25, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=712, fps=20, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=712, fps=15, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=712, fps=10, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=712, fps=7.5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=712, fps=5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=30, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=25, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=20, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=15, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=10, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=7.5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=30, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=25, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=20, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=15, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=10, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=7.5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=2560, height=720, fps=30, fourcc='MJPG', stereo=True, is_default=False, is_supported=True), CaptureMode(width=2560, height=720, fps=25, fourcc='MJPG', stereo=True, is_default=False, is_supported=True), CaptureMode(width=2560, height=720, fps=20, fourcc='MJPG', stereo=True, is_default=False, is_supported=True), CaptureMode(width=2560, height=720, fps=15, fourcc='MJPG', stereo=True, is_default=False, is_supported=True), CaptureMode(width=2560, height=720, fps=10, fourcc='MJPG', stereo=True, is_default=False, is_supported=True), CaptureMode(width=2560, height=720, fps=7.5, fourcc='MJPG', stereo=True, is_default=False, is_supported=True), CaptureMode(width=2560, height=720, fps=5, fourcc='MJPG', stereo=True, is_default=False, is_supported=True), CaptureMode(width=1280, height=480, fps=30, fourcc='MJPG', stereo=True, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=25, fourcc='MJPG', stereo=True, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=20, fourcc='MJPG', stereo=True, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=15, fourcc='MJPG', stereo=True, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=10, fourcc='MJPG', stereo=True, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=7.5, fourcc='MJPG', stereo=True, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=5, fourcc='MJPG', stereo=True, is_default=False, is_supported=False), CaptureMode(width=1280, height=360, fps=30, fourcc='MJPG', stereo=True, is_default=True, is_supported=True), CaptureMode(width=1280, height=360, fps=25, fourcc='MJPG', stereo=True, is_default=False, is_supported=True), CaptureMode(width=1280, height=360, fps=20, fourcc='MJPG', stereo=True, is_default=False, is_supported=True), CaptureMode(width=1280, height=360, fps=15, fourcc='MJPG', stereo=True, is_default=False, is_supported=True), CaptureMode(width=1280, height=360, fps=10, fourcc='MJPG', stereo=True, is_default=False, is_supported=True), CaptureMode(width=1280, height=360, fps=7.5, fourcc='MJPG', stereo=True, is_default=False, is_supported=True), CaptureMode(width=1280, height=360, fps=5, fourcc='MJPG', stereo=True, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=10, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=7.5, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=5, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=480, fps=30, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=25, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=20, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=15, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=10, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=7.5, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=5, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=360, fps=30, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=25, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=20, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=15, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=10, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=7.5, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=5, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=712, fps=10, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=712, fps=7.5, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=712, fps=5, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=30, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=25, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=20, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=15, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=10, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=7.5, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=5, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=30, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=25, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=20, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=15, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=10, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=7.5, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=5, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=2560, height=720, fps=5, fourcc='YUYV', stereo=True, is_default=False, is_supported=True), CaptureMode(width=1280, height=480, fps=15, fourcc='YUYV', stereo=True, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=10, fourcc='YUYV', stereo=True, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=7.5, fourcc='YUYV', stereo=True, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=5, fourcc='YUYV', stereo=True, is_default=False, is_supported=False), CaptureMode(width=1280, height=360, fps=25, fourcc='YUYV', stereo=True, is_default=False, is_supported=True), CaptureMode(width=1280, height=360, fps=20, fourcc='YUYV', stereo=True, is_default=False, is_supported=True), CaptureMode(width=1280, height=360, fps=15, fourcc='YUYV', stereo=True, is_default=False, is_supported=True), CaptureMode(width=1280, height=360, fps=10, fourcc='YUYV', stereo=True, is_default=False, is_supported=True), CaptureMode(width=1280, height=360, fps=7.5, fourcc='YUYV', stereo=True, is_default=False, is_supported=True), CaptureMode(width=1280, height=360, fps=5, fourcc='YUYV', stereo=True, is_default=False, is_supported=True)), controls=(CameraControl(kind='enum', value_type='int', default=3, label='Auto exposure', min=0, max=3, step=1, enum_values=((1, 'Manual'), (3, 'Aperture Priority')), unit='', name='auto_exposure', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=312, label='Exposure', min=1, max=80000, step=1, enum_values=(), unit='100us', name='exposure', current_value=None, auto_companion='auto_exposure', updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=32, label='Gain', min=0, max=130, step=1, enum_values=(), unit='', name='gain', current_value=None, auto_companion='auto_exposure', updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=0, label='Brightness', min=-64, max=64, step=1, enum_values=(), unit='', name='brightness', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=34, label='Contrast', min=0, max=64, step=1, enum_values=(), unit='', name='contrast', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=62, label='Saturation', min=0, max=78, step=1, enum_values=(), unit='', name='saturation', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=0, label='Hue', min=-2000, max=2000, step=1, enum_values=(), unit='', name='hue', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=150, label='Gamma', min=100, max=300, step=1, enum_values=(), unit='', name='gamma', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=4, label='Sharpness', min=1, max=7, step=1, enum_values=(), unit='', name='sharpness', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='boolean', value_type='bool', default=1, label='Auto white balance', min=0, max=1, step=1, enum_values=(), unit='', name='white_balance_auto', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=4600, label='White balance (K)', min=2800, max=6500, step=1, enum_values=(), unit='K', name='white_balance', current_value=None, auto_companion='white_balance_auto', updatable_by_auto=False, writable=True), CameraControl(kind='enum', value_type='int', default=1, label='Power-line frequency', min=0, max=2, step=1, enum_values=((0, 'Disabled'), (1, '50 Hz'), (2, '60 Hz')), unit='', name='power_line_frequency', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=0, label='Backlight compensation', min=0, max=1, step=1, enum_values=(), unit='', name='backlight_compensation', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True)), windows_controls=(CameraControl(kind='enum', value_type='int', default=3, label='Auto exposure', min=0, max=3, step=1, enum_values=((1, 'Manual'), (3, 'Aperture Priority')), unit='', name='auto_exposure', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=-5, label='Exposure', min=-13, max=3, step=1, enum_values=(), unit='log2(s)', name='exposure', current_value=None, auto_companion='auto_exposure', updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=32, label='Gain', min=0, max=130, step=1, enum_values=(), unit='', name='gain', current_value=None, auto_companion='auto_exposure', updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=0, label='Brightness', min=-64, max=64, step=1, enum_values=(), unit='', name='brightness', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=34, label='Contrast', min=0, max=64, step=1, enum_values=(), unit='', name='contrast', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=62, label='Saturation', min=0, max=78, step=1, enum_values=(), unit='', name='saturation', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=0, label='Hue', min=-2000, max=2000, step=1, enum_values=(), unit='', name='hue', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=150, label='Gamma', min=100, max=300, step=1, enum_values=(), unit='', name='gamma', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=4, label='Sharpness', min=1, max=7, step=1, enum_values=(), unit='', name='sharpness', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='boolean', value_type='bool', default=1, label='Auto white balance', min=0, max=1, step=1, enum_values=(), unit='', name='white_balance_auto', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=4600, label='White balance (K)', min=2800, max=6500, step=1, enum_values=(), unit='K', name='white_balance', current_value=None, auto_companion='white_balance_auto', updatable_by_auto=False, writable=True), CameraControl(kind='number', value_type='int', default=0, label='Backlight compensation', min=0, max=1, step=1, enum_values=(), unit='', name='backlight_compensation', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True)), skus=('STEREO-CAMERA-MINIMA-REGULAR-LENSES-V21', 'STEREO-CAMERA-MINIMA-WIDE-LENSES-V21', 'STEREO-CAMERA-MINIMA-ULTRA-WIDE-LENSES-V21', 'STEREO-CAMERA-MINIMA-FISHEYE-LENSES-V21', 'STEREO-CAMERA-MODULE-MINIMA-REGULAR-LENSES-NO-MOUNT-V21', 'STEREO-CAMERA-MODULE-MINIMA-REGULAR-LENSES-SIDE-MOUNTS-V21', 'STEREO-CAMERA-MODULE-MINIMA-WIDE-LENSES-NO-MOUNT-V21', 'STEREO-CAMERA-MODULE-MINIMA-WIDE-LENSES-SIDE-MOUNTS-V21', 'STEREO-CAMERA-MODULE-MINIMA-ULTRA-WIDE-LENSES-NO-MOUNT-V21', 'STEREO-CAMERA-MODULE-MINIMA-ULTRA-WIDE-LENSES-SIDE-MOUNTS-V21', 'STEREO-CAMERA-MODULE-MINIMA-FISHEYE-LENSES-NO-MOUNT-V21', 'STEREO-CAMERA-MODULE-MINIMA-FISHEYE-LENSES-SIDE-MOUNTS-V21')), CameraSpec(modes=(CaptureMode(width=2560, height=720, fps=30, fourcc='MJPG', stereo=True, is_default=True, is_supported=True), CaptureMode(width=2560, height=720, fps=25, fourcc='MJPG', stereo=True, is_default=False, is_supported=True), CaptureMode(width=2560, height=720, fps=20, fourcc='MJPG', stereo=True, is_default=False, is_supported=True), CaptureMode(width=2560, height=720, fps=15, fourcc='MJPG', stereo=True, is_default=False, is_supported=True), CaptureMode(width=2560, height=720, fps=10, fourcc='MJPG', stereo=True, is_default=False, is_supported=True), CaptureMode(width=2560, height=720, fps=7.5, fourcc='MJPG', stereo=True, is_default=False, is_supported=True), CaptureMode(width=2560, height=720, fps=5, fourcc='MJPG', stereo=True, is_default=False, is_supported=True), CaptureMode(width=3840, height=1080, fps=30, fourcc='MJPG', stereo=True, is_default=False, is_supported=True), CaptureMode(width=3840, height=1080, fps=25, fourcc='MJPG', stereo=True, is_default=False, is_supported=True), CaptureMode(width=3840, height=1080, fps=20, fourcc='MJPG', stereo=True, is_default=False, is_supported=True), CaptureMode(width=3840, height=1080, fps=15, fourcc='MJPG', stereo=True, is_default=False, is_supported=True), CaptureMode(width=3840, height=1080, fps=10, fourcc='MJPG', stereo=True, is_default=False, is_supported=True), CaptureMode(width=3840, height=1080, fps=7.5, fourcc='MJPG', stereo=True, is_default=False, is_supported=True), CaptureMode(width=3840, height=1080, fps=5, fourcc='MJPG', stereo=True, is_default=False, is_supported=True), CaptureMode(width=1920, height=1080, fps=30, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1920, height=1080, fps=25, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1920, height=1080, fps=20, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1920, height=1080, fps=15, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1920, height=1080, fps=10, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1920, height=1080, fps=7.5, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1920, height=1080, fps=5, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=30, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=25, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=20, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=15, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=10, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=7.5, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=5, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1600, height=600, fps=30, fourcc='MJPG', stereo=True, is_default=False, is_supported=False), CaptureMode(width=1600, height=600, fps=25, fourcc='MJPG', stereo=True, is_default=False, is_supported=False), CaptureMode(width=1600, height=600, fps=20, fourcc='MJPG', stereo=True, is_default=False, is_supported=False), CaptureMode(width=1600, height=600, fps=15, fourcc='MJPG', stereo=True, is_default=False, is_supported=False), CaptureMode(width=1600, height=600, fps=10, fourcc='MJPG', stereo=True, is_default=False, is_supported=False), CaptureMode(width=1600, height=600, fps=7.5, fourcc='MJPG', stereo=True, is_default=False, is_supported=False), CaptureMode(width=1600, height=600, fps=5, fourcc='MJPG', stereo=True, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=30, fourcc='MJPG', stereo=True, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=25, fourcc='MJPG', stereo=True, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=20, fourcc='MJPG', stereo=True, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=15, fourcc='MJPG', stereo=True, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=10, fourcc='MJPG', stereo=True, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=7.5, fourcc='MJPG', stereo=True, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=5, fourcc='MJPG', stereo=True, is_default=False, is_supported=False), CaptureMode(width=640, height=240, fps=30, fourcc='MJPG', stereo=True, is_default=False, is_supported=False), CaptureMode(width=640, height=240, fps=25, fourcc='MJPG', stereo=True, is_default=False, is_supported=False), CaptureMode(width=640, height=240, fps=20, fourcc='MJPG', stereo=True, is_default=False, is_supported=False), CaptureMode(width=640, height=240, fps=15, fourcc='MJPG', stereo=True, is_default=False, is_supported=False), CaptureMode(width=640, height=240, fps=10, fourcc='MJPG', stereo=True, is_default=False, is_supported=False), CaptureMode(width=640, height=240, fps=7.5, fourcc='MJPG', stereo=True, is_default=False, is_supported=False), CaptureMode(width=640, height=240, fps=5, fourcc='MJPG', stereo=True, is_default=False, is_supported=False), CaptureMode(width=800, height=600, fps=30, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=800, height=600, fps=25, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=800, height=600, fps=20, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=800, height=600, fps=15, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=800, height=600, fps=10, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=800, height=600, fps=7.5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=800, height=600, fps=5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=30, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=25, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=20, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=15, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=10, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=7.5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=320, height=240, fps=30, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=320, height=240, fps=25, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=320, height=240, fps=20, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=320, height=240, fps=15, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=320, height=240, fps=10, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=320, height=240, fps=7.5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=320, height=240, fps=5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1920, height=1072, fps=30, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1920, height=1072, fps=25, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1920, height=1072, fps=20, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1920, height=1072, fps=15, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1920, height=1072, fps=10, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1920, height=1072, fps=7.5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1920, height=1072, fps=5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=712, fps=30, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=712, fps=25, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=712, fps=20, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=712, fps=15, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=712, fps=10, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=712, fps=7.5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=712, fps=5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=800, height=592, fps=30, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=800, height=592, fps=25, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=800, height=592, fps=20, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=800, height=592, fps=15, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=800, height=592, fps=10, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=800, height=592, fps=7.5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=800, height=592, fps=5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=30, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=25, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=20, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=15, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=10, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=7.5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=320, height=232, fps=30, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=320, height=232, fps=25, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=320, height=232, fps=20, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=320, height=232, fps=15, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=320, height=232, fps=10, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=320, height=232, fps=7.5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=320, height=232, fps=5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=3840, height=1080, fps=2.5, fourcc='YUYV', stereo=True, is_default=False, is_supported=True), CaptureMode(width=2560, height=720, fps=5, fourcc='YUYV', stereo=True, is_default=False, is_supported=True), CaptureMode(width=2560, height=720, fps=2.5, fourcc='YUYV', stereo=True, is_default=False, is_supported=True), CaptureMode(width=1920, height=1080, fps=5, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1920, height=1080, fps=2.5, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=7.5, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=5, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1600, height=600, fps=10, fourcc='YUYV', stereo=True, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=20, fourcc='YUYV', stereo=True, is_default=False, is_supported=False), CaptureMode(width=640, height=240, fps=30, fourcc='YUYV', stereo=True, is_default=False, is_supported=False), CaptureMode(width=800, height=600, fps=20, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=30, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=320, height=240, fps=30, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1920, height=1072, fps=5, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=712, fps=7.5, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=800, height=592, fps=20, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=30, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=320, height=232, fps=30, fourcc='YUYV', stereo=False, is_default=False, is_supported=False)), controls=(CameraControl(kind='enum', value_type='int', default=3, label='Auto exposure', min=0, max=3, step=1, enum_values=((1, 'Manual'), (3, 'Aperture Priority')), unit='', name='auto_exposure', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=312, label='Exposure', min=1, max=80000, step=1, enum_values=(), unit='100us', name='exposure', current_value=None, auto_companion='auto_exposure', updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=32, label='Gain', min=0, max=130, step=1, enum_values=(), unit='', name='gain', current_value=None, auto_companion='auto_exposure', updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=0, label='Brightness', min=-64, max=64, step=1, enum_values=(), unit='', name='brightness', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=34, label='Contrast', min=0, max=64, step=1, enum_values=(), unit='', name='contrast', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=62, label='Saturation', min=0, max=78, step=1, enum_values=(), unit='', name='saturation', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=0, label='Hue', min=-2000, max=2000, step=1, enum_values=(), unit='', name='hue', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=150, label='Gamma', min=100, max=300, step=1, enum_values=(), unit='', name='gamma', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=4, label='Sharpness', min=1, max=7, step=1, enum_values=(), unit='', name='sharpness', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='boolean', value_type='bool', default=1, label='Auto white balance', min=0, max=1, step=1, enum_values=(), unit='', name='white_balance_auto', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=4600, label='White balance (K)', min=2800, max=6500, step=1, enum_values=(), unit='K', name='white_balance', current_value=None, auto_companion='white_balance_auto', updatable_by_auto=False, writable=True), CameraControl(kind='enum', value_type='int', default=1, label='Power-line frequency', min=0, max=2, step=1, enum_values=((0, 'Disabled'), (1, '50 Hz'), (2, '60 Hz')), unit='', name='power_line_frequency', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=0, label='Backlight compensation', min=0, max=1, step=1, enum_values=(), unit='', name='backlight_compensation', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True)), windows_controls=(CameraControl(kind='enum', value_type='int', default=3, label='Auto exposure', min=0, max=3, step=1, enum_values=((1, 'Manual'), (3, 'Aperture Priority')), unit='', name='auto_exposure', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=-5, label='Exposure', min=-13, max=3, step=1, enum_values=(), unit='log2(s)', name='exposure', current_value=None, auto_companion='auto_exposure', updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=32, label='Gain', min=0, max=130, step=1, enum_values=(), unit='', name='gain', current_value=None, auto_companion='auto_exposure', updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=0, label='Brightness', min=-64, max=64, step=1, enum_values=(), unit='', name='brightness', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=34, label='Contrast', min=0, max=64, step=1, enum_values=(), unit='', name='contrast', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=62, label='Saturation', min=0, max=78, step=1, enum_values=(), unit='', name='saturation', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=0, label='Hue', min=-2000, max=2000, step=1, enum_values=(), unit='', name='hue', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=150, label='Gamma', min=100, max=300, step=1, enum_values=(), unit='', name='gamma', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=4, label='Sharpness', min=1, max=7, step=1, enum_values=(), unit='', name='sharpness', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='boolean', value_type='bool', default=1, label='Auto white balance', min=0, max=1, step=1, enum_values=(), unit='', name='white_balance_auto', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=4600, label='White balance (K)', min=2800, max=6500, step=1, enum_values=(), unit='K', name='white_balance', current_value=None, auto_companion='white_balance_auto', updatable_by_auto=False, writable=True), CameraControl(kind='number', value_type='int', default=0, label='Backlight compensation', min=0, max=1, step=1, enum_values=(), unit='', name='backlight_compensation', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True)), skus=('STEREO-CAMERA-CLARITY-REGULAR-LENSES-V21', 'STEREO-CAMERA-CLARITY-WIDE-LENSES-V21', 'STEREO-CAMERA-CLARITY-ULTRA-WIDE-LENSES-V21', 'STEREO-CAMERA-CLARITY-FISHEYE-LENSES-V21', 'STEREO-CAMERA-CLARITY-ULTRA-FISHEYE-LENSES-V21', 'STEREO-CAMERA-MODULE-CLARITY-REGULAR-LENSES-V21', 'STEREO-CAMERA-MODULE-CLARITY-WIDE-LENSES-V21', 'STEREO-CAMERA-MODULE-CLARITY-ULTRA-WIDE-LENSES-V21', 'STEREO-CAMERA-MODULE-CLARITY-FISHEYE-LENSES-V21', 'STEREO-CAMERA-MODULE-CLARITY-ULTRA-FISHEYE-LENSES-V21')), CameraSpec(modes=(CaptureMode(width=1280, height=720, fps=30, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=25, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=20, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=15, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=10, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=7.5, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=5, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=480, fps=30, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=25, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=20, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=15, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=10, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=7.5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=360, fps=30, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=25, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=20, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=15, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=10, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=7.5, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=5, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=712, fps=30, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=712, fps=25, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=712, fps=20, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=712, fps=15, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=712, fps=10, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=712, fps=7.5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=712, fps=5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=30, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=25, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=20, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=15, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=10, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=7.5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=30, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=25, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=20, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=15, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=10, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=7.5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=2560, height=720, fps=30, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=2560, height=720, fps=25, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=2560, height=720, fps=20, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=2560, height=720, fps=15, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=2560, height=720, fps=10, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=2560, height=720, fps=7.5, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=2560, height=720, fps=5, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=480, fps=30, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=25, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=20, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=15, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=10, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=7.5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=360, fps=30, fourcc='MJPG', stereo=False, is_default=True, is_supported=True), CaptureMode(width=1280, height=360, fps=25, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=360, fps=20, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=360, fps=15, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=360, fps=10, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=360, fps=7.5, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=360, fps=5, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=10, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=7.5, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=5, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=480, fps=30, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=25, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=20, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=15, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=10, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=7.5, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=5, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=360, fps=30, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=25, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=20, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=15, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=10, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=7.5, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=5, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=712, fps=10, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=712, fps=7.5, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=712, fps=5, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=30, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=25, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=20, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=15, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=10, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=7.5, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=5, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=30, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=25, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=20, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=15, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=10, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=7.5, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=5, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=2560, height=720, fps=5, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=480, fps=15, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=10, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=7.5, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=5, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=360, fps=25, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=360, fps=20, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=360, fps=15, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=360, fps=10, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=360, fps=7.5, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=360, fps=5, fourcc='YUYV', stereo=False, is_default=False, is_supported=True)), controls=(CameraControl(kind='enum', value_type='int', default=3, label='Auto exposure', min=0, max=3, step=1, enum_values=((1, 'Manual'), (3, 'Aperture Priority')), unit='', name='auto_exposure', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=312, label='Exposure', min=1, max=80000, step=1, enum_values=(), unit='100us', name='exposure', current_value=None, auto_companion='auto_exposure', updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=32, label='Gain', min=0, max=130, step=1, enum_values=(), unit='', name='gain', current_value=None, auto_companion='auto_exposure', updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=0, label='Brightness', min=-64, max=64, step=1, enum_values=(), unit='', name='brightness', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=34, label='Contrast', min=0, max=64, step=1, enum_values=(), unit='', name='contrast', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=62, label='Saturation', min=0, max=78, step=1, enum_values=(), unit='', name='saturation', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=0, label='Hue', min=-2000, max=2000, step=1, enum_values=(), unit='', name='hue', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=150, label='Gamma', min=100, max=300, step=1, enum_values=(), unit='', name='gamma', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=4, label='Sharpness', min=1, max=7, step=1, enum_values=(), unit='', name='sharpness', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='boolean', value_type='bool', default=1, label='Auto white balance', min=0, max=1, step=1, enum_values=(), unit='', name='white_balance_auto', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=4600, label='White balance (K)', min=2800, max=6500, step=1, enum_values=(), unit='K', name='white_balance', current_value=None, auto_companion='white_balance_auto', updatable_by_auto=False, writable=True), CameraControl(kind='enum', value_type='int', default=1, label='Power-line frequency', min=0, max=2, step=1, enum_values=((0, 'Disabled'), (1, '50 Hz'), (2, '60 Hz')), unit='', name='power_line_frequency', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=0, label='Backlight compensation', min=0, max=1, step=1, enum_values=(), unit='', name='backlight_compensation', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True)), windows_controls=(CameraControl(kind='enum', value_type='int', default=3, label='Auto exposure', min=0, max=3, step=1, enum_values=((1, 'Manual'), (3, 'Aperture Priority')), unit='', name='auto_exposure', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=-5, label='Exposure', min=-13, max=3, step=1, enum_values=(), unit='log2(s)', name='exposure', current_value=None, auto_companion='auto_exposure', updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=32, label='Gain', min=0, max=130, step=1, enum_values=(), unit='', name='gain', current_value=None, auto_companion='auto_exposure', updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=0, label='Brightness', min=-64, max=64, step=1, enum_values=(), unit='', name='brightness', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=34, label='Contrast', min=0, max=64, step=1, enum_values=(), unit='', name='contrast', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=62, label='Saturation', min=0, max=78, step=1, enum_values=(), unit='', name='saturation', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=0, label='Hue', min=-2000, max=2000, step=1, enum_values=(), unit='', name='hue', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=150, label='Gamma', min=100, max=300, step=1, enum_values=(), unit='', name='gamma', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=4, label='Sharpness', min=1, max=7, step=1, enum_values=(), unit='', name='sharpness', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='boolean', value_type='bool', default=1, label='Auto white balance', min=0, max=1, step=1, enum_values=(), unit='', name='white_balance_auto', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=4600, label='White balance (K)', min=2800, max=6500, step=1, enum_values=(), unit='K', name='white_balance', current_value=None, auto_companion='white_balance_auto', updatable_by_auto=False, writable=True), CameraControl(kind='number', value_type='int', default=0, label='Backlight compensation', min=0, max=1, step=1, enum_values=(), unit='', name='backlight_compensation', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True)), skus=('CAMERA-MONO-REGULAR',)), CameraSpec(modes=(CaptureMode(width=1280, height=720, fps=30, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=25, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=20, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=15, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=10, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=7.5, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=5, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=480, fps=30, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=25, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=20, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=15, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=10, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=7.5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=360, fps=30, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=25, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=20, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=15, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=10, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=7.5, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=5, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=712, fps=30, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=712, fps=25, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=712, fps=20, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=712, fps=15, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=712, fps=10, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=712, fps=7.5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=712, fps=5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=30, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=25, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=20, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=15, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=10, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=7.5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=30, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=25, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=20, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=15, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=10, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=7.5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=2560, height=720, fps=30, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=2560, height=720, fps=25, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=2560, height=720, fps=20, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=2560, height=720, fps=15, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=2560, height=720, fps=10, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=2560, height=720, fps=7.5, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=2560, height=720, fps=5, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=480, fps=30, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=25, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=20, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=15, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=10, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=7.5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=5, fourcc='MJPG', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=360, fps=30, fourcc='MJPG', stereo=False, is_default=True, is_supported=True), CaptureMode(width=1280, height=360, fps=25, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=360, fps=20, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=360, fps=15, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=360, fps=10, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=360, fps=7.5, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=360, fps=5, fourcc='MJPG', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=10, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=7.5, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=720, fps=5, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=480, fps=30, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=25, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=20, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=15, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=10, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=7.5, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=480, fps=5, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=360, fps=30, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=25, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=20, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=15, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=10, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=7.5, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=640, height=360, fps=5, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=712, fps=10, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=712, fps=7.5, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=712, fps=5, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=30, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=25, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=20, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=15, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=10, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=7.5, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=472, fps=5, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=30, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=25, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=20, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=15, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=10, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=7.5, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=640, height=352, fps=5, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=2560, height=720, fps=5, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=480, fps=15, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=10, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=7.5, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=480, fps=5, fourcc='YUYV', stereo=False, is_default=False, is_supported=False), CaptureMode(width=1280, height=360, fps=25, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=360, fps=20, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=360, fps=15, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=360, fps=10, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=360, fps=7.5, fourcc='YUYV', stereo=False, is_default=False, is_supported=True), CaptureMode(width=1280, height=360, fps=5, fourcc='YUYV', stereo=False, is_default=False, is_supported=True)), controls=(CameraControl(kind='enum', value_type='int', default=3, label='Auto exposure', min=0, max=3, step=1, enum_values=((1, 'Manual'), (3, 'Aperture Priority')), unit='', name='auto_exposure', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=312, label='Exposure', min=1, max=80000, step=1, enum_values=(), unit='100us', name='exposure', current_value=None, auto_companion='auto_exposure', updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=32, label='Gain', min=0, max=130, step=1, enum_values=(), unit='', name='gain', current_value=None, auto_companion='auto_exposure', updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=0, label='Brightness', min=-64, max=64, step=1, enum_values=(), unit='', name='brightness', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=34, label='Contrast', min=0, max=64, step=1, enum_values=(), unit='', name='contrast', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=62, label='Saturation', min=0, max=78, step=1, enum_values=(), unit='', name='saturation', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=0, label='Hue', min=-2000, max=2000, step=1, enum_values=(), unit='', name='hue', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=150, label='Gamma', min=100, max=300, step=1, enum_values=(), unit='', name='gamma', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=4, label='Sharpness', min=1, max=7, step=1, enum_values=(), unit='', name='sharpness', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='boolean', value_type='bool', default=1, label='Auto white balance', min=0, max=1, step=1, enum_values=(), unit='', name='white_balance_auto', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=4600, label='White balance (K)', min=2800, max=6500, step=1, enum_values=(), unit='K', name='white_balance', current_value=None, auto_companion='white_balance_auto', updatable_by_auto=False, writable=True), CameraControl(kind='enum', value_type='int', default=1, label='Power-line frequency', min=0, max=2, step=1, enum_values=((0, 'Disabled'), (1, '50 Hz'), (2, '60 Hz')), unit='', name='power_line_frequency', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=0, label='Backlight compensation', min=0, max=1, step=1, enum_values=(), unit='', name='backlight_compensation', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True)), windows_controls=(CameraControl(kind='enum', value_type='int', default=3, label='Auto exposure', min=0, max=3, step=1, enum_values=((1, 'Manual'), (3, 'Aperture Priority')), unit='', name='auto_exposure', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=-5, label='Exposure', min=-13, max=3, step=1, enum_values=(), unit='log2(s)', name='exposure', current_value=None, auto_companion='auto_exposure', updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=32, label='Gain', min=0, max=130, step=1, enum_values=(), unit='', name='gain', current_value=None, auto_companion='auto_exposure', updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=0, label='Brightness', min=-64, max=64, step=1, enum_values=(), unit='', name='brightness', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=34, label='Contrast', min=0, max=64, step=1, enum_values=(), unit='', name='contrast', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=62, label='Saturation', min=0, max=78, step=1, enum_values=(), unit='', name='saturation', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=0, label='Hue', min=-2000, max=2000, step=1, enum_values=(), unit='', name='hue', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=150, label='Gamma', min=100, max=300, step=1, enum_values=(), unit='', name='gamma', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=4, label='Sharpness', min=1, max=7, step=1, enum_values=(), unit='', name='sharpness', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='boolean', value_type='bool', default=1, label='Auto white balance', min=0, max=1, step=1, enum_values=(), unit='', name='white_balance_auto', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True), CameraControl(kind='number', value_type='int', default=4600, label='White balance (K)', min=2800, max=6500, step=1, enum_values=(), unit='K', name='white_balance', current_value=None, auto_companion='white_balance_auto', updatable_by_auto=False, writable=True), CameraControl(kind='number', value_type='int', default=0, label='Backlight compensation', min=0, max=1, step=1, enum_values=(), unit='', name='backlight_compensation', current_value=None, auto_companion=None, updatable_by_auto=True, writable=True)), skus=('CAMERA-MONO-MINI-REGULAR',)))) -> Optional[depz_camera_sdk.cameras.camera_spec.CameraSpec]
```

Return the spec whose :attr:`CameraSpec.skus` contains ``sku``.

The single detection path: the server's camera-info endpoint
returns a SKU for a given serial; this function maps that to the
corresponding :class:`CameraSpec`. ``None`` if no spec claims
that SKU.

### list_connected_depz_cameras_detailed

```python
list_connected_depz_cameras_detailed() -> tuple[list[depz_camera_sdk.cameras.camera_info.DepzCameraInfo], list[dict]]
```

Like :func:`list_connected_depz_cameras`, but failures become DATA
instead of silently shrinking the list.

Returns ``(resolved, failed)`` where each ``failed`` entry is
``{"serial", "os_id", "error"}`` for a physically-attached DEPZ
device whose device-info fetch / SKU resolution / calibration parse
failed. UI layers use this to show a degraded entry ("camera seen
but info fetch failed") rather than pretending the camera isn't
plugged in.

## Platform

### PlatformBackend

```python
class PlatformBackend()
```

Abstraction over OS-specific camera enumeration + streaming
setup. Concrete backends declare :attr:`av_input_format` and
:attr:`clock_domain` as ``ClassVar[str]``.

#### PlatformBackend.list_cameras

```python
list_cameras(self) -> list[depz_camera_sdk.platform.base.VideoDeviceInfo]
```

Enumerate DEPZ cameras attached to the host. Implementations
SHOULD filter to DEPZ-only by parsing each camera's USB
iProduct via :func:`parse_depz_iproduct`.

#### PlatformBackend.open_controls

```python
open_controls(self, os_id: str, serial: Optional[str] = None) -> depz_camera_sdk.platform.base.CameraControls
```

Open a control handle for the camera identifier returned by
:meth:`list_cameras`.

``serial`` is the camera's USB serial (the
:attr:`VideoDeviceInfo.serial` parsed from iProduct). Backends
whose ``os_id`` doesn't uniquely identify the USB device use it
to bind to the exact unit when several DEPZ cameras are attached
(macOS: the AVFoundation index can't be mapped to a USB device,
so the serial is the only disambiguator). ``None`` falls back to
the first matching DEPZ camera.

#### PlatformBackend.av_open_args

```python
av_open_args(self, os_id: str, mode: 'CaptureMode') -> tuple[str, str, dict]
```

Build the ``(url, format, options)`` triple for :func:`av.open`
to start a stream of ``mode`` from the camera ``os_id``.

``mode`` exposes ``.width`` / ``.height`` / ``.fps`` / ``.fourcc``.
``options`` is the complete, demuxer-specific options dict — the
caller passes it to :func:`av.open` verbatim and adds nothing.
Linux builds a v4l2 option set (``input_format`` key); Windows
builds a dshow option set (``pixel_format`` / ``vcodec`` /
``video_device_number``). Raises :class:`ValueError` for a fourcc
the backend can't map.

#### PlatformBackend.detect_fourcc

```python
detect_fourcc(self, mode_fourcc: str, codec_context: 'av.codec.context.CodecContext') -> str
```

Return the effective fourcc for frames coming from this device.

The default implementation trusts ``mode_fourcc`` (from
:class:`CaptureMode`).  Backends where the device may deliver a
different pixel layout than requested (e.g.  AVFoundation delivers
raw yuyv422 for MJPEG modes) override this to inspect
``codec_context`` and return the actual wire format.

``mode_fourcc`` is always uppercase (``"MJPG"``, ``"YUYV"``);
the return value must also be uppercase and match one of the
branches in :meth:`FrameSource.decode_to_bgr`.

#### PlatformBackend.trim_payload

```python
trim_payload(self, payload: bytes, width: int, height: int) -> bytes
```

Trim raw frame payload to exactly ``width * height * 2`` bytes.

AVFoundation appends a padding line to raw YUYV frames; other
backends deliver frames whose payload already matches the
expected size.  Override this in backends that need stripping.

### CameraControls

```python
class CameraControls()
```

Per-camera handle for reading/writing camera controls (exposure,
gain, brightness, ...). Distinct from the streaming source — uses a
parallel handle (V4L2 second fd / cv2 with CAP_*).

Range/default/availability come from the matched
:class:`~depz_camera_sdk.cameras.camera_spec.CameraSpec` (single source of
truth — one curated table per firmware class). The platform handle
is a thin pipe for live get/set only.

#### CameraControls.get

```python
get(self, name: str) -> Optional[float]
```

Return current value of a canonical-name control, or None if
not available on this device.

#### CameraControls.set

```python
set(self, name: str, value: float) -> bool
```

Apply value. Returns True on success, False if not writable.

#### CameraControls.close

```python
close(self) -> None
```

### VideoDeviceInfo

```python
class VideoDeviceInfo(os_id: str, name: str, serial: Optional[str]) -> None
```

Discovery result for one DEPZ camera attached to the host.

Returned by :meth:`PlatformBackend.list_cameras`. The
``os_id`` is the opaque string the backend will
accept again in :meth:`PlatformBackend.open_controls` and
:meth:`PlatformBackend.os_id_to_av_url` — not
portable across OSes (concrete shape lives in the per-backend
docstring).

### current_backend

```python
current_backend() -> depz_camera_sdk.platform.base.PlatformBackend
```

Return the singleton backend for the current OS.

Lazily imports the concrete backend module so the wrong platform's
module (with its OS-only deps) is never touched.

## Controls

### Control

```python
class Control(kind: str, value_type: str, default: Any = None, label: str = '', min: float | None = None, max: float | None = None, step: float | None = None, enum_values: tuple[tuple[int | float | str, str], ...] = (), unit: str = '', name: str = '') -> None
```

Self-description of one runtime control — pure data (fields +
validation + :meth:`to_dict`).

Declare controls on a host via the typed subclasses (:class:`IntControl`
etc.), which add read-only descriptor behaviour. ``Control`` itself is
the shared base, and the plain data shape that camera-control specs
(:class:`~depz_camera_sdk.cameras.camera_spec.CameraControl`)
subclass directly.

#### Control.to_dict

```python
to_dict(self) -> dict
```

## Cache

### get_cache_dir

```python
get_cache_dir() -> pathlib.Path
```

Return the active SDK cache root. See module docstring for
resolution order. Caller is responsible for creating the directory
if it doesn't exist (it's lazy).

### set_cache_dir

```python
set_cache_dir(path: Union[str, pathlib.Path]) -> None
```

Override the SDK-default cache root for the current process.
Hosting apps (e.g. the web viewer) call this at startup to redirect
cache into a platform-standard application data dir. Pass an
absolute path; ``~``-prefix expanded automatically.

## Logging_Setup

### configure_logging

```python
configure_logging(*, component: str, level: Union[str, int] = 'INFO', log_dir: Union[str, pathlib.Path, NoneType] = None, session: Optional[str] = None, console: bool = True) -> pathlib.Path
```

Configure root logging for this process. Idempotent — the first
call wins; later calls return the active session dir unchanged.

``component`` names this process's log file (``"server"``,
``"pipeline-host"`` — pid-suffixed, ``"sdk"``). ``level`` sets the
CONSOLE threshold; the file handler always records DEBUG. ``session``
lets a child process join its parent's session directory (see
:func:`get_child_log_config`).

Returns the session directory in use. If the log directory cannot be
created (read-only install, exotic sandbox), logging degrades to
console-only rather than failing startup — that failure itself is
logged.

### enable_default_logging

```python
enable_default_logging(level: Union[str, int] = 'INFO') -> pathlib.Path
```

Opt-in convenience for SDK-standalone users::

    import depz_camera_sdk
    depz_camera_sdk.enable_default_logging()

Console + ``<log_root>/<session>/sdk.log``. Returns the session dir.
Nothing is configured at import time — this call is the opt-in.
