# depz-camera-sdk — guide

A hand-written explanation of how the SDK is shaped and how to use it.
For exact signatures, read the docstrings in `depz_camera_sdk/` —
they are the source of truth.

## Contents

- [What it is](#what-it-is)
- [Installation](#installation)
- [Mental model](#mental-model)
- [Hello-world (the shortest thing that runs)](#hello-world-the-shortest-thing-that-runs)
- [I want to…](#i-want-to)
- [Using the SDK](#using-the-sdk)
  - [Sources](#sources)
  - [Models — the inference / variant catalog](#models--the-inference--variant-catalog)
  - [Inference engines](#inference-engines)
  - [Calibration](#calibration)
  - [Recording](#recording)
  - [Camera controls (live only)](#camera-controls-live-only)
  - [Stream toggling](#stream-toggling)
  - [Filters](#filters)
  - [Seek (replay only)](#seek-replay-only)
  - [Pipeline-level controls](#pipeline-level-controls)
  - [Metrics](#metrics)
- [Extending the SDK](#extending-the-sdk)
  - [Running your own stereo / mono depth model](#running-your-own-stereo--mono-depth-model)
  - [Custom sources](#custom-sources)
  - [Adding a new filter](#adding-a-new-filter)
  - [Adding a new lens model](#adding-a-new-lens-model)
- [Appendix](#appendix)
  - [Source argument forms](#source-argument-forms)
  - [Calibration wire format](#calibration-wire-format)
- [Common pitfalls](#common-pitfalls)
- [Class reference](#class-reference)

---

## What it is

`depz-camera-sdk` turns a stereo (side-by-side) or mono
DEPZ camera — or a previously-recorded MKV file — into a stream of
`FramePacket`s, frame-numbered bundles of raw eyes, rectified eyes,
disparity, depth, and a 3D point cloud.

Four collaborators do the heavy lifting; you build them and hand them
to a `DepthPipeline`:

1. **`FrameSource`** — yields raw camera packets (MJPG / YUYV).
   Two built-ins: `CameraSource` (live UVC) and `MkvSource` (replay).
2. **`Calibration`** — rectifies the eye pair onto the inference
   canvas and projects disparity → depth → point cloud. Always comes
   from the source.
3. **`DepthInferenceBase`** — the disparity-or-depth solver. One ABC,
   three intermediate bases (`NeuralStereoInferenceBase`,
   `NeuralMonoInferenceBase`, `ClassicStereoInferenceBase`), and a
   set of concrete inferences, each its own class:
   - `LightStereoInference`, `CREStereoInitialInference`,
     `CREStereoCombinedInference`, `CFastFoundationStereoInference` —
     neural stereo nets producing canvas-pixel **disparity**
     (preprocess + postprocess + the `.onnx` weights + the runtime
     engine).
   - `DepthAnythingV2SmallInference`, `Metric3DV2VitSmallInference` —
     neural monocular nets producing **depth** directly. Two runtime
     controls (`depth_scale` / `depth_offset`) calibrate relative output
     to a reference distance.
   - `SgbmInference` / `StereoBMInference` — classical cv2 block
     matchers. Eleven runtime controls each, mutable via
     `pipeline.set_inference_control(name, value)`.
   - `RectifyOnlyInference` — no-op for rectification-only pipelines.
4. **`DepthPipeline`** — orchestrator. Single worker thread + a
   thread-safe command queue for runtime mutations. Build it, call
   `start()`, pull packets off `get()`.

You don't normally touch the inner stages — pick a source, pick an
inference, hand both to a pipeline.

---

## Installation

The SDK ships on PyPI as `depz-camera-sdk` for Windows x64, Linux x86_64,
Linux ARM64 (aarch64) and macOS 11+ (Intel + Apple Silicon, one universal2 wheel), any Python ≥ 3.11
(stable-ABI wheels). A bare install is "pure" — no inference engines at all;
add extras for your hardware. `[openvino]` covers Intel CPU/iGPU/NPU and
AMD CPU (Windows/Linux x86_64 only). The ONNX Runtime extras are mutually
exclusive — pick at most one:

```bash
pip install depz-camera-sdk                       # pure: no engines
pip install depz-camera-sdk[openvino]             # the x86_64 default
pip install depz-camera-sdk[openvino,cuda-bundled] # + NVIDIA GPU, self-contained (~1.5 GB)
pip install depz-camera-sdk[openvino,cuda]        # + NVIDIA GPU, you provide the CUDA 12 runtime
pip install depz-camera-sdk[openvino,dml]         # + AMD GPU on Windows (any DX12 GPU)
pip install depz-camera-sdk[openvino,migraphx]    # + AMD GPU on Linux (system ROCm + MIGraphX)
pip install depz-camera-sdk[cpu]                  # Linux ARM64 / macOS baseline
```

On macOS there is no CUDA/DirectML/OpenVINO: `[cpu]` installs the stock
ONNX Runtime wheel, which on Apple Silicon includes the CoreML EP
(GPU/ANE) — and everything else in this guide (live capture, recording,
replay, all inferences) works the same as on Linux and Windows.

If you want the browser viewer rather than the raw SDK, either
`pip install depz-camera-viewer[cpu]` (same extras) or use the no-Python
one-click installers from
[depz-camera-sdk](https://github.com/depz-ai/depz-camera-sdk/releases/latest).

---

## Mental model

### Stage flow

The pipeline is a single worker thread that drives one frame per
iteration through five stages:

```
                         DepthPipeline (worker thread)
   ┌───────────────────────────────────────────────────────────────────────┐
   │  (1) grab         (2) decode/rectify    (3) inference    (4) post-    │
   │     │                 │                     │                process  │
   │     ▼                 ▼                     ▼                  ▼      │
   │  FrameSource ──► Calibration ──► DepthInferenceBase ──► Postprocessor │
   │  (jpeg+pts_us)   (canvas pair)   (disp / depth)         (filters,     │
   │                                                          depth,       │
   │                                                          cloud)       │
   │                                                               │       │
   │                                                               ▼       │
   │                                                          FramePacket ─┼─► output queue
   └───────────────────────────────────────────────────────────────────────┘
                                     ▲
                            ┌────────┴────────┐
                            │ MkvRecorder     │   (optional, attached
                            │ (writer thread) │    via RecordingController)
                            └─────────────────┘
```

Stage by stage:

- ① **`FrameSource.grab()`** — produces a `RawFrame` (jpeg/yuyv
  bytes + `pts_us` source-relative microseconds). Live cameras run a
  pump thread internally; replay paces from disk.
- ② **`Calibration.canvas_remap_pair()`** — decodes wire bytes to
  BGR, splits SBS halves (stereo) or passes through (mono), then
  rectifies onto the inference canvas in one `cv2.remap` per eye.
- ③ **`DepthInferenceBase.submit()`** — for neural backends, composes
  architecture-specific preprocess (BGR → NCHW float32) with the
  engine's async `submit` and returns a future. The worker
  preprocesses the next frame while N−1 inferences are in flight
  (`pipeline_depth` defaults to 2). Classical backends compute
  synchronously and hand back a pre-completed `EventFuture` — the
  polling loop is uniform.
- ④ **`Postprocessor.process()`** — `inference.postprocess` (squeeze
  + crop) → optional disparity filters → `disparity_to_depth` (only
  on the disparity branch) → depth filters → `depth_to_point_cloud`
  → cloud filters → `FramePacket`.
- The `FramePacket` lands on a bounded output queue;
  `pipeline.get(timeout=...)` is the consumer side.

**Branch on `inference.OUTPUTS_DEPTH`.** Stereo backends emit
disparity in canvas pixels; mono backends emit depth (× scale +
offset). The postprocessor takes the disparity → depth conversion
out of the path on the depth branch — and `FramePacket.disparity` is
`None`.

**Filters live in `Postprocessor`**, not in the inference. Each
filter declares `OPERATES_ON ∈ {"disparity", "depth", "point_cloud"}`
and the postprocessor partitions them around the projection steps.

### Threading

The worker thread is the **only** writer of pipeline state
(calibration, filter config, stream-enable map, source position,
pipeline controls). RPC / GUI threads mutate by enqueuing `_Command`
objects; the worker drains them between iterations. Public methods
like `set_calibration_control`, `set_filter_control`, `seek`, `pause`
are safe to call from any thread without explicit locking.

Camera-control writes (`set_camera_control`) bypass the command
queue intentionally — they hit a separate fd / ioctl, no race.
Recording fanout runs on the source's pump thread.

---

## Hello-world (the shortest thing that runs)

```python
from depz_camera_sdk import DepthPipeline

for frame in DepthPipeline():
    print(f"frame {frame.frame_number}, depth shape {frame.depth.shape}")
```

### Five-minute, fully-explicit form

```python
from depz_camera_sdk import (
    DepthPipeline, CameraSource, list_connected_depz_cameras,
)
from depz_camera_sdk.inference import build_inference

camera_info  = list_connected_depz_cameras()[0]
capture_mode = camera_info.spec.get_mode("1280x720@30/MJPG/stereo")
camera_source = CameraSource(camera_info, capture_mode)

inference = build_inference("lightstereo", "s-sceneflow-opt-256x512",
                            engine_slug="openvino-CPU")

pipeline = DepthPipeline(
    source=camera_source,
    inference=inference,
    enable_raw=True,           # frame.left / frame.right
    enable_rectified=True,     # frame.rectified_left / .rectified_right
    enable_disparity=True,     # frame.disparity
    enable_depth=True,         # frame.depth
    enable_cloud=True,         # frame.point_cloud
)
for frame in pipeline:
    ...
```

### Classical (no GPU/NPU, no model file)

`build_inference` works for classical inferences too — they set
`NEURAL = False`, declare no files, and `build_inference` constructs
the concrete `SgbmInference` / `StereoBMInference` directly with the
variant's `canvas_size`.

```python
inference = build_inference("sgbm", "360x640")   # returns SgbmInference

with DepthPipeline(source=src, inference=inference) as pipe:
    pipe.set_inference_control("num_disparities", 128)
    pipe.set_inference_control("block_size", 7)
    # ... pull packets ...
```

If you'd rather construct the classical inference directly (no
catalog), the bare `SgbmInference(canvas_size=(360, 640))` /
`StereoBMInference(canvas_size=(360, 640))` form still works.

### Monocular depth (single-eye / stereo run on left eye)

```python
inference = build_inference("depth-anything-v2-small", "518x518")   # returns DepthAnythingV2SmallInference

with DepthPipeline(source=src, inference=inference) as pipe:
    pipe.set_inference_control("depth_scale", 2.5)   # tune to meters
    pipe.set_inference_control("depth_offset", 0.1)
```

`build_inference` constructs the concrete inference class registered
for the variant's slug — no need for the caller to pick stereo vs mono.
The class itself carries `OUTPUTS_DEPTH` so the pipeline routes the
output correctly.

### Recording, in two lines

```python
pipe.start_recording("/tmp/session.mkv")
# ... let it run ...
result = pipe.stop_recording()
print(f"{result.frames} frames, {result.bytes} bytes, {result.duration_s:.1f}s")
```

The pipeline derives the embedded `camera_info.json` automatically
from `source.depz_info` (live) or the input MKV's attachment (when
re-recording from replay). No sidecar files — calibration history,
camera-prop snapshots, and pipeline config are all embedded as MKV
attachments.

## I want to…

| I want to… | Read |
| --- | --- |
| Read depth from a live camera | [Hello-world](#hello-world-the-shortest-thing-that-runs), [Sources](#sources) |
| Replay a recording, seek, step frame-by-frame | [Sources → MkvSource](#mkvsource--replay), [Seek](#seek-replay-only) |
| Record to MKV (and replay it later) | [Recording](#recording) |
| Pick / download a neural model | [Models](#models--the-inference--variant-catalog) |
| Run on GPU / choose the runtime | [Inference engines](#inference-engines) |
| Use my own `.onnx` file | [Bring your own .onnx](#bring-your-own-onnx-file) |
| Run my own stereo / mono depth model | [Running your own model](#running-your-own-stereo--mono-depth-model) |
| Tune SGBM / filters / streams at runtime | [Filters](#filters), [Stream toggling](#stream-toggling), [Pipeline-level controls](#pipeline-level-controls) |
| Drive camera exposure / gain / white balance | [Camera controls](#camera-controls-live-only) |
| See where time is going | [Metrics](#metrics) |
| Plug in my own source / filter / lens model | [Extending the SDK](#extending-the-sdk) |
| Look up an exact signature | [api.md](api.md) |

---

## Using the SDK

Task-oriented sections, one per subsystem. Each stands alone —
jump to what you need.

### Sources

#### `CameraSource` — live capture

```python
from depz_camera_sdk import CameraSource, list_connected_depz_cameras

cam  = list_connected_depz_cameras()[0]

src  = CameraSource(cam)                                # spec's default mode (omit `mode=`)
src  = CameraSource(cam, cam.spec.default_mode)         # same, explicit
src  = CameraSource(cam, cam.spec.get_mode(             # pick a specific slug
            "1280x720@30/MJPG/stereo"))                 # "{eye_w}x{eye_h}@{fps}/{FOURCC}[/stereo]"
```

`CameraSource` requires a resolved `DepzCameraInfo` — the SDK
doesn't drive raw / unrecognized USB cameras. Capture modes,
controls, and calibration must all be known up front, which is
exactly what `DepzCameraInfo` carries. Resolve via
`list_connected_depz_cameras()` /
`get_connected_depz_camera_by_serial()`, or build one yourself with
`DepzCameraInfo.from_dict(payload, os_id=...)` if you have a
server-shape payload on hand.

Internally runs a **pump thread** that reads from V4L2 / DShow at
full rate and hands the latest frame to the pipeline (drop-to-latest
— older frames inference didn't pick up are discarded). `grab()`
blocks on a condition variable, so there's no wasteful polling.

The bound `DepzCameraInfo` lets `DepthPipeline.start_recording`
derive the `camera_info.json` MKV attachment automatically.

#### `MkvSource` — replay

```python
from depz_camera_sdk import MkvSource

src = MkvSource("/path/to/recording.mkv",
                replay_mode="realtime",    # "every_frame" = block on consumer
                loop=False,
                build_frame_index=True)    # default; needed for step API
src.pause(); src.resume()
src.seek_relative_us(5_000_000)            # 5 s into the file
src.step_one_forward()                     # exact ±1 frame (paused only)
```

The pipeline mirrors the same API — `pipeline.pause()` / `resume()` /
`is_pausable` / `is_paused` / `step_replay("forward"|"backward")` —
so application code rarely needs to touch the source directly.

`MkvSource` extracts calibration + camera-props snapshot embedded by
the recorder. The `active_calibration` property is **stateful**: it
returns the live `Calibration` for the currently-active replay
timestamp, so online-recal patches stored in the recording's
`calibration_history.jsonl` attachment are honoured during playback.

Every form the `DepthPipeline` ctor accepts (serial string, `.mkv`
path, `.json` path, …) is listed in the appendix under
[Source argument forms](#source-argument-forms). To plug in your own
source class, see [Custom sources](#custom-sources).

### Models — the inference / variant catalog

Neural inferences need an `.onnx` file on disk. The SDK ships a
**catalog**: every concrete inference class is registered via
`@register_inference("slug")`; each class advertises one or more
`InferenceVariant`s (size × dataset × resolution × iter-count). The
catalog handles:

- listing what's available without hitting the network,
- resolving variant slugs to download URLs (one tiny HTTP call,
  cached per process),
- downloading + sha256-verifying weights into a local cache,
- atomic writes (no half-downloaded files on Ctrl-C),
- handing back a ready-to-use `DepthInferenceBase` for the pipeline.

**No models are bundled in the wheel.** First use fetches on demand.

#### Where models live on disk

By default models cache under
`./depz_sdk_cache/models/<inference>/<variant>/<file>` in the current
working directory. Hosting apps redirect this once at startup:

```python
from depz_camera_sdk import set_cache_dir

set_cache_dir("~/.cache/depz")                  # Linux
# set_cache_dir("~/Library/Caches/depz")        # macOS
# set_cache_dir(r"%LOCALAPPDATA%\depz\Cache")   # Windows
```

After that, every component (engines, model downloader, camera-info
store) writes under `<cache>/...`.

#### Walking the catalog

```python
from depz_camera_sdk.inference import available_inferences

for cls in available_inferences():
    domain = "depth" if cls.OUTPUTS_DEPTH else "disparity"
    fam = f"  [{cls.ARCHITECTURE_FAMILY}]" if cls.ARCHITECTURE_FAMILY else ""
    print(f"{cls.SLUG} — {cls.LABEL} ({domain}){fam}")
    for v in cls.variants():
        mark = "✓ cached" if v.is_cached else "  needs download"
        print(f"   {v.slug:<32} {v.label}   {mark}")
```

Typical printout (truncated — CREStereo Combined alone has 20 variants):

```
c-fast-foundationstereo — C-Fast-FoundationStereo (disparity)  [FoundationStereo]
   iters8-576x960                      iters 8 / 576×960                    needs download
   iters4-576x960                      iters 4 / 576×960                    needs download
crestereo-combined — CREStereo Combined (disparity)  [CREStereo]
   iter2-240x320                       iter 2 / 240×320                     needs download
   iter2-320x480                       iter 2 / 320×480                     needs download
   ...
crestereo-initial — CREStereo Initial (disparity)  [CREStereo]
   iter2-160x240                       iter 2 / 160×240                   ✓ cached
   iter2-180x320                       iter 2 / 180×320                     needs download
   ...
depth-anything-v2-small — Depth Anything V2 Small (depth)
   518x518                             518×518 (default)                    needs download
lightstereo — LightStereo (disparity)
   s-sceneflow-opt-256x512             S / SceneFlow / 256×512 (opt)      ✓ cached
   s-sceneflow-opt-576x960             S / SceneFlow / 576×960 (opt)        needs download
   s-sceneflow-origin-256x512          S / SceneFlow / 256×512 (origin)     needs download
lite-any-stereo — Lite Any Stereo (disparity)
   v2-s-384x640                        V2 S / 384×640                       needs download
metric3d-v2-vit-small — Metric3D V2 (ViT-Small) (depth)
   vit-small-504x504                   ViT-Small / 504×504                  needs download
rectify-only — Rectify only (no inference) (disparity)
   256x512                             256×512 (fast)                     ✓ cached  (no download)
   ...
sgbm — SGBM (classical, CPU) (disparity)
   256x512                             256×512 (fast)                     ✓ cached  (no download)
   360x640                             360×640 (balanced)                 ✓ cached  (no download)
   480x720                             480×720 (detailed)                 ✓ cached  (no download)
stereo-bm — StereoBM (classical, CPU, fastest) (disparity)
   256x512                             256×512 (fast)                     ✓ cached  (no download)
   ...
```

`ARCHITECTURE_FAMILY` is an optional grouping tag. The two CREStereo
inferences (`crestereo-initial`, `crestereo-combined`) ship as
separate classes but share the upstream architecture, so both set
`ARCHITECTURE_FAMILY = "CREStereo"` — the UI can collapse them under
one heading. Inferences without a sibling leave it as `None`.

#### Building an inference from a catalog variant

The one-liner. `build_inference` walks the steps below internally —
this is what the hello-world uses.

```python
from depz_camera_sdk.inference import build_inference

inference = build_inference(
    "lightstereo", "s-sceneflow-opt-256x512",
    download_progress_cb=lambda done, total: print(f"\r{done}/{total}", end=""),
)
```

If you want to do it by hand (testing one stage in isolation):

```python
from depz_camera_sdk import recommend_engine
from depz_camera_sdk.engines import make_engine
from depz_camera_sdk.inference import get_inference

cls     = get_inference("lightstereo")
variant = cls.get_variant("s-sceneflow-opt-256x512")

# 1. Ensure cached + return the absolute paths of every declared
#    file. files[0] is the entry-point (model.onnx); siblings
#    (model.onnx_data for ONNX with external data) follow in
#    declaration order.
files = variant.resolve_files()

# 2. Build the engine around the entry-point file. ``engine_kwargs``
#    on the variant carry per-variant engine controls (``input_shape``
#    for dynamic-shape ONNXes, etc.).
engine = make_engine(recommend_engine(), str(files[0]),
                     **variant.engine_kwargs)

# 3. Construct the inference. ``inference_kwargs`` on the variant
#    carry per-variant ctor args (empty for LightStereo; canvas_size
#    for SGBM/StereoBM/RectifyOnly).
inference = cls(engine=engine, **variant.inference_kwargs)
```

Classical inferences (`sgbm`, `stereo-bm`, `rectify-only`) set
`NEURAL = False`; `build_inference` then skips file resolution +
engine building entirely and just calls `cls(**variant.inference_kwargs)`.
You can construct them directly too: `SgbmInference(canvas_size=(360, 640))`.

#### Bring your own .onnx file

If you already have a `.onnx` on disk (training output, custom
fine-tune, hand-downloaded asset), skip the catalog and construct
the concrete inference class directly with a built engine:

```python
from depz_camera_sdk import recommend_engine
from depz_camera_sdk.engines import make_engine
from depz_camera_sdk.inference import (
    LightStereoInference,
    DepthAnythingV2SmallInference,
)

# Stereo (disparity).
inf = LightStereoInference(
    engine=make_engine("openvino-CPU", "/abs/path/to/my-model.onnx"),
)

# Mono (depth). Concrete class is the source of truth for
# preprocess/postprocess + OUTPUTS_DEPTH; pick the one that matches
# your .onnx export.
inf = DepthAnythingV2SmallInference(
    engine=make_engine(recommend_engine(),
                       "/abs/path/to/some-model.onnx"),
)
```

No filename-based architecture detection, no slug-driven dispatch
fallback — the caller picks the inference class explicitly and builds
the engine themselves.

#### Pre-warming the cache (CI, offline machines, kiosks)

```python
from depz_camera_sdk.inference import available_inferences

for cls in available_inferences():
    for v in cls.variants():
        if not v.files_to_download:
            continue
        v.resolve_files()           # downloads + verifies; ignore the paths
```

Combine with `set_cache_dir(...)` to redirect into whatever read-only
image / squashfs / NFS share you ship.

### Inference engines

Two backends (OpenVINO + ONNX Runtime) with the same async
interface, identified by **slug strings** like `"openvino-CPU"`,
`"onnx-cuda"`:

```python
from depz_camera_sdk import available_engines, recommend_engine

for slug in available_engines():
    print(slug)
# openvino-GPU
# openvino-CPU
# onnx-cuda
# onnx-cpu

print(recommend_engine())   # → "openvino-GPU" on this host (best-first)
```

`available_engines()` is sorted best-first. `recommend_engine()`
returns `available_engines()[0]` (or `None` if no backend loads).

You normally pass the slug straight into a neural inference
constructor. To construct a raw engine object (rare — testing,
custom pipelines), use the lower-level factory:

```python
from depz_camera_sdk.engines import make_engine, parse_engine_slug
engine = make_engine("openvino-CPU", "model.onnx",
                     cache_dir=None, do_models_cache=True, nireq=2)
parse_engine_slug("openvino-CPU")   # → ("openvino", "CPU")
```

**Common engine kwargs** (forwarded via `engine_kwargs={...}` to
neural inferences, or directly to `make_engine`):

- `cache_dir` — override the SDK-wide cache root (defaults to
  `get_cache_dir()`); compiled-model cache lives at
  `<cache_dir>/<backend>/`.
- `do_models_cache` (OpenVINO) — set False to disable kernel-graph
  cache entirely (read-only filesystems / one-shot scripts).
- `nireq` (OpenVINO) — async-queue depth.
- `providers` (ONNX Runtime) — provider preference list override.
- `input_shape` — pin a `(1, 3, H, W)` shape on dynamic-input
  ONNXes. Most inferences already export with a fixed input;
  Metric3D V2 et al. set `INPUT_SIZE` on the inference class and
  `build_inference` wires this for you.

OpenVINO uses `AsyncInferQueue`; ONNX Runtime uses a small thread
pool plus `session.run()`. Either way, the pipeline never blocks on
inference — it submits and pulls completed futures FIFO. First GPU
compile is ~30–60 s; subsequent loads are sub-second thanks to the
on-disk cache.

### Calibration

Two concrete classes ship today: `OpenCVBasicCalibration` (pinhole +
8-coefficient radtan) and `OpenCVFisheyeCalibration` (Kannala–
Brandt fisheye, lon/lat-rectified). Both register against the same
wire format and share all common math; only their distortion
coefficients and rectification kernel differ.

You normally don't construct calibrations yourself — the source
hands one to the pipeline. When you do need to parse a payload:

```python
from depz_camera_sdk import Calibration

calib = Calibration.from_dict(payload)            # device-info shape
calib.set_param("rectify_alpha", 0.0)
patch_dict = calib.to_dict()                      # round-trip
```

`Calibration.from_dict` dispatches on the cereal `polymorphic_name`
field — no `kind` tag is consumed or emitted; identification is
structural.

#### Rescaling

The pipeline rescales calibration automatically when capture eye
size differs by a uniform factor (e.g. 2× or 0.5× of the native
calibration size). Non-uniform rescale is rejected — re-calibrate.

#### Fisheye notes

`OpenCVFisheyeCalibration.disparity_to_depth(disp)` returns **slant
range** (Euclidean distance from camera origin to the 3D point), not
z-forward depth. Spherical scenes carry a single radial distance per
pixel; flattening to z-forward loses meaning at large azimuth /
elevation. `depth_to_point_cloud` handles this correctly, but
downstream consumers reading `FramePacket.depth` should be aware.

The full JSON wire format lives in the appendix under
[Calibration wire format](#calibration-wire-format); adding a new
lens model is covered in
[Extending the SDK](#adding-a-new-lens-model).

### Recording

Records the wire bytes verbatim — no re-encode. Output is a
Matroska file with the calibration, pipeline config, and
camera-props snapshot embedded as **MKV attachments** (no sidecar
files):

```python
pipe.start_recording("/tmp/session.mkv")
# ... let it run ...
result = pipe.stop_recording()
print(result.frames, result.bytes, result.duration_s, result.dropped)
```

The `camera_info.json` attachment that goes into the MKV is sourced
automatically:

| Source kind | Where the payload comes from |
| --- | --- |
| `CameraSource` | `source.depz_info.to_dict()` — server-shape, identical to the device-info endpoint response. |
| `MkvSource` (re-recording from replay) | The `camera_info.json` attachment of the input MKV, embedded verbatim. |
| Custom source / mock | Reconstructed from `pipeline.active_calibration.to_dict()` plus identity off `pipeline.camera_info`. |

A live camera fans out **every** captured frame to the recorder via
the source's pump thread (so you don't lose frames if inference
can't keep up). Replay sources fall back to feeding at worker rate —
fine since replay is bounded.

YUYV captures are written as `rawvideo` (FOURCC tag `YUY2`). MJPG
is remuxed as-is. Camera-prop snapshots (one per second) and
calibration patches are buffered in memory during recording and
embedded as the `camera_state_history.jsonl` and
`calibration_history.jsonl` attachments at `stop_recording()` time.
Re-mux uses `ffmpeg` from `$PATH`.

A recording can also be paused and resumed without closing the
file: `pipe.pause_recording()` stops writing frames while keeping
the output open, `pipe.resume_recording()` continues into the same
MKV. Both are no-ops when nothing is being recorded.

### Camera controls (live only)

Three call sites, all on the pipeline:

```python
pipe.set_camera_control("exposure", 312)         # V4L2 100us units
v = pipe.get_camera_control("exposure")          # → 312.0

controls = pipe.list_camera_controls()            # → list[CameraControl]
for c in controls:
    print(c.name, c.label, c.kind, c.current_value, c.min, c.max,
          c.unit, c.auto_companion, c.updatable_by_auto)
```

`list_camera_controls()` returns a snapshot of every control the
underlying camera actually exposes. Each entry is a public
`CameraControl` — internal V4L2 / Win32 IDs are stripped. Replay
sources return `[]`.

**What's in each `CameraControl`:**

- **Identity**: `name` (canonical id), `label` (UI string).
- **Type**: `kind ∈ {"number", "enum", "boolean"}`, `enum_values`
  (for enums), `unit` (`"100us"`, `"K"`, `"%"`, …).
- **Live state**: `current_value`, `min`, `max`, `step`, `default`.
  Driver-authoritative on Linux; spec literals as fallback.
- **Gating**: `auto_companion` — name of the AUTO/MANUAL toggle that
  controls whether *this* ctrl is currently writable. `None` for
  ungated controls. `updatable_by_auto` — whether readback is
  meaningful while gated.
- **Writability**: `writable` — `False` for read-only telemetry
  (rare).

The control fd is parallel to the streaming fd, so writes don't
block the pipeline. `set_camera_control` bypasses the command
queue intentionally — different fd, different ioctl, no race.

### Stream toggling

By default the pipeline computes a depth-only output. To save CPU
or expose more streams, flip them on/off at runtime:

```python
pipe.set_stream_enabled("cloud",     True)    # back-projection on
pipe.set_stream_enabled("disparity", True)    # expose raw disparity
pipe.set_stream_enabled("raw",       False)   # skip eye .copy()
```

Dependencies are enforced. `cloud` depends on `depth` which depends
on `disparity`. Enabling `cloud` cascades upstream automatically.
Disabling `disparity` while `depth`/`cloud` are still on **fails**
with a descriptive reason — caller must disable downstream first.

```python
applied, reason = pipe.set_stream_enabled("disparity", False)
if not applied:
    print("refused:", reason)        # "depended on by: depth"
```

### Filters

Six filters in the default chain. Each is its own self-describing
class subclassing one of `DisparityFilter`, `DepthFilter`,
`PointCloudFilter` (which pre-fill `OPERATES_ON`). The pipeline
applies them in construction order, partitioning by `OPERATES_ON`:

| Order | Filter | Slug | Domain | Stateful? |
|---|---|---|---|---|
| 1 | `SpeckleFilter`             | `speckle`       | disparity   | no |
| — | *disparity → depth* (cal.)  | —               | —           | — |
| 2 | `ZClipFilter`               | `zclip`         | depth       | no |
| 3 | `EdgeFilter`                | `edge`          | depth       | scratch buffers |
| 4 | `TemporalEmaFilter`         | `temporal_ema`  | depth       | prev frame |
| — | *depth → cloud* (cal.)      | —               | —           | — |
| 5 | `BoundingBoxFilter`         | `bbox`          | point_cloud | no |
| 6 | `StatisticalOutlierFilter`  | `stat_outlier`  | point_cloud | no |

Each filter declares its own canonical id via `Filter.SLUG`; RPC +
UI use the slug for routing.

**Mutating parameters:** address each filter by its slug.

```python
pipe.set_filter_control("speckle",      "enabled", True)
pipe.set_filter_control("speckle",      "max_size", 80)      # clamped
pipe.set_filter_control("temporal_ema", "enabled", True)
pipe.set_filter_control("temporal_ema", "alpha",   0.7)      # clamped
pipe.set_filter_control("stat_outlier", "enabled", True)
pipe.set_filter_control("stat_outlier", "std_ratio", 1.5)
```

Each call returns the value actually stored after coercion +
clamping. Goes through the worker's command queue.

**Introspection** (drives the web viewer's UI panel without any
hardcoded control list):

```python
pipe.list_filters()
# → [
#   {"slug": "speckle", "label": "Speckle (disparity)",
#    "operates_on": "disparity",
#    "parameters": [
#       {"name": "enabled", "kind": "boolean", "value_type": "bool",
#        "default": False, ...},
#       {"name": "max_size", "kind": "number", "value_type": "int",
#        "default": 50, "min": 1, "max": 2000, "step": 1, ...},
#    ],
#    "state": {"enabled": True, "max_size": 80, "max_diff": 1.0}},
#   ...
# ]
```

Each parameter descriptor carries a `kind` (UI widget hint —
`number` / `enum` / `boolean`) and a `value_type` (the data type —
`int` / `float` / `bool` / `str`); the two are independent.

**Reset semantics.** Every filter's `reset()` is called by the
pipeline on `seek()`, `set_calibration_control()` /
`update_calibration()` (including
`set_calibration_control("canvas_fit_mode", ...)`), and on a
disable→enable flip. Stateless filters use the inherited no-op
default.

`ZClipFilter` vs the web viewer's render-side `min_depth` /
`max_depth`: zclip is **destructive** (zeroes pixels server-side,
before depth → cloud projection); render-side min/max only changes
colormap normalization and cloud render culling. Both useful,
distinct.

Writing your own filter is covered in
[Extending the SDK](#adding-a-new-filter).

### Seek (replay only)

```python
pipe.seek(fraction=0.5)                # 50 % of duration
pipe.seek(relative_us=10_000_000)      # 10 s in
pipe.seek(monotonic_ns=t_capture_ns)   # absolute monotonic — for IMU sync
pipe.seek(utc_ns=int(time.time()*1e9)) # absolute UTC
```

`seek()` is **synchronous**. Internally it drains all in-flight
inference, flushes the output queue, resets all stateful filters,
then asks the source to seek. When it returns, the system is in a
clean state — no half-rendered frame from the old position will
appear.

Frame numbers reset to 0 after a seek; `pkt.pts_us` continues to
reflect microseconds since stream start (source-local). To get
absolute monotonic-clock time of a packet, use
`pipeline.pts_to_monotonic_ns(pkt.pts_us)` — returns int
nanoseconds, bit-exact, no float rounding.

### Pipeline-level controls

Mirroring the filter / inference / calibration descriptor pattern,
`DepthPipeline` exposes a small set of pipeline-wide runtime controls.

```python
pipe.pipeline_info()
# → {"controls": [<descriptor>, ...],
#    "state":    {"photo_mode": False,
#                 "output_drop_policy": "drop_oldest",
#                 "replay_calibration_mode": "all"}}

pipe.set_pipeline_control("photo_mode", True)
pipe.set_pipeline_control("output_drop_policy", "block")
```

Built-ins:

- **`photo_mode`** (boolean) — bypass inference every frame; only
  run it on `pipe.trigger_snapshot()`. The resulting packet has
  `is_snapshot=True` and is what `pipe.get_snapshot()` returns.
  Useful for capture stations where the live preview shouldn't burn
  GPU.
- **`output_drop_policy`** (enum: `drop_oldest` | `block`) — overflow
  behaviour of the output queue. `drop_oldest` for live UI
  (freshness > completeness), `block` for offline replay analysis
  (lossless).
- **`replay_calibration_mode`** (enum: `all` | `exclude_controls` |
  `none`) — replay-only. Controls how recorded calibration patches
  apply during playback. `all` = honour every patch verbatim;
  `exclude_controls` = apply patches but preserve the user's current
  `rectify_alpha` / `canvas_fit_mode` / FoVs; `none` = ignore the
  history entirely.

Stream-graph gates (`raw` / `rectified` / `disparity` / `depth` /
`cloud`) are NOT on this list — their cascade semantics live in
`set_stream_enabled()`. Construction-time arguments
(`pipeline_depth`, `max_output_queue`) are also not here — they're
allocation parameters, fixed at start.

### Metrics

```python
m = pipe.metrics
print(f"camera {m.camera_fps:.1f} / inference {m.inference_fps:.1f} fps")
print(f"worker p50 {m.worker_loop_p50_us/1000:.1f} ms / "
      f"p95 {m.worker_loop_p95_us/1000:.1f} ms")
print(f"camera drops {m.camera_drops} (rate {m.camera_drops_per_s:.1f}/s), "
      f"output drops {m.output_drops} (rate {m.output_drops_per_s:.1f}/s)")
print(f"in flight: {m.inference_in_flight}")
```

`PipelineMetrics` is a frozen dataclass — cheap to snapshot from
any thread. Counters reset on `seek()`. Includes per-stage p50
breakdowns (`decode_p50_us`, `rectify_p50_us`, `inference_p50_us`,
`postprocess_p50_us`) so you can localise where time is going.

---

## Extending the SDK

Every extension point follows the same shape: subclass a base,
declare yourself, register (or append to a chain) — the pipeline,
RPC layer, and web UI pick the new piece up without further wiring.

### Running your own stereo / mono depth model

Two situations, one decision point — does your `.onnx` match the
preprocess/postprocess contract of an architecture the SDK already
ships?

**Same architecture (a fine-tune / re-export).** No subclassing:
construct the shipped inference class around your file and hand it to
a pipeline — see [Bring your own .onnx file](#bring-your-own-onnx-file).

**New architecture.** Subclass one of the two neural bases —
`NeuralStereoInferenceBase` (model emits **disparity**) or
`NeuralMonoInferenceBase` (model emits **depth**; inherits the
`depth_scale` / `depth_offset` affine controls — call
`self._apply_depth_affine(depth)` at the tail of your postprocess) —
and implement the two per-network adapter methods:

- `preprocess(canvas_l, canvas_r)` — BGR uint8 canvases → tuple of
  NCHW float32 blobs, in the order the model declares its inputs.
  Mono networks receive `canvas_r=None` and return a single-blob
  tuple.
- `postprocess(raw_output, canvas_layout)` — raw engine output → 2D
  float32 map sliced to the unpadded canvas region
  (`canvas_layout.pad_x/pad_y/valid_w/valid_h`). Stereo returns
  disparity in canvas pixels; mono returns depth.

```python
import cv2
import numpy as np

from depz_camera_sdk import DepthPipeline, recommend_engine
from depz_camera_sdk.engines import make_engine
from depz_camera_sdk.inference import NeuralStereoInferenceBase


class MyStereoNet(NeuralStereoInferenceBase):
    SLUG = "my-stereo-net"

    def preprocess(self, canvas_l, canvas_r):
        blob = lambda img: cv2.dnn.blobFromImage(
            img, scalefactor=1 / 255.0, swapRB=True)
        return blob(canvas_l), blob(canvas_r)

    def postprocess(self, raw_output, canvas_layout):
        disp = np.squeeze(np.asarray(raw_output)).astype(np.float32)
        y, x = canvas_layout.pad_y, canvas_layout.pad_x
        h, w = canvas_layout.valid_h, canvas_layout.valid_w
        return np.ascontiguousarray(disp[y:y + h, x:x + w])


inf = MyStereoNet(engine=make_engine(recommend_engine(),
                                     "/abs/path/to/my-model.onnx"))

with DepthPipeline(source=src, inference=inf, enable_cloud=True) as pipe:
    pkt = pipe.get(timeout=10.0)
    print(pkt.depth.shape, pkt.point_cloud.shape)
```

The pipeline takes the instance as-is — rectification onto the
model's input canvas, disparity → depth → point-cloud projection,
filters, recording, and the web-viewer wiring all come for free.
Registering in the catalog (`@register_inference` + `variants()`) is
only needed if you also want the model listed / downloadable via
`build_inference`; a direct instance like above skips that entirely.

### Custom sources

Both built-ins extend `FrameSource`. To plug in your own (e.g. a
network stream): inherit `FrameSource`, implement
`open()` / `grab()` / `close()` plus the relevant capability
properties. Defaults already cover most things — sane no-ops for
`add_fanout`, `controls`, `seek_relative_us`, etc. If your bytes are
MJPG or YUYV, the default `decode_to_bgr` works.

### Adding a new filter

Subclass one of `DisparityFilter` /
`DepthFilter` / `PointCloudFilter`, set `SLUG` + optional `LABEL`,
declare your controls as typed-control class attributes (`BoolControl` /
`IntControl` / `FloatControl` / `StrControl` — the first one MUST be an
`enabled` `BoolControl`), implement `apply()`, then append to
`default_filter_chain()` (in `depz_camera_sdk/filters/__init__.py`).
Read-only properties, default seeding, and the inherited `controls()`
accessor all come from the control declarations — no boilerplate. The web UI
surfaces the new panel automatically — no JS changes.

```python
class MyFilter(DepthFilter):
    SLUG = "my_filter"
    enabled  = BoolControl(default=False, label="Enable")
    strength = FloatControl("number", default=0.5,
                         min=0.0, max=1.0, step=0.05, label="Strength")

    def apply(self, frame):
        if not self.enabled:          # generated read-only property
            return frame
        ...                           # use self.strength
```

### Adding a new lens model

Subclass `Calibration`, set `POLYMORPHIC_NAME` and `SDK_PARAMS`,
implement `_extract_distortion`, `_distortion_params_dict`,
`_compute_rectification`, `set_param`, then decorate with
`@register_calibration`. The base does the rest. Use
`registered_polymorphic_names()` to introspect what's been wired up.

---

## Appendix

### Source argument forms

The pipeline ctor delegates every form below to one helper —
`resolve_source_arg(source) -> FrameSource` (in
`depz_camera_sdk.sources`). You can call it yourself when you want
the same dispatch outside a pipeline (CLI tools, tests):

| Form | Meaning |
| --- | --- |
| `FrameSource` instance | Returned as-is. Use this when you want to construct `CameraSource(cam, mode=...)` with a non-default mode, or pass a custom subclass. |
| `DepzCameraInfo` | Builds `CameraSource(cam, cam.spec.default_mode)`. Skip-the-string-roundtrip shortcut when you already have one from `list_connected_depz_cameras()`. |
| Bare DEPZ serial string (`"RRX6L1PSJ4"`) | Looks up the live camera by serial, returns `CameraSource` at the spec's default mode. |
| `*.mkv` path | Replay via `MkvSource`. |
| `*.json` path | A server-shape camera-info file. The SDK reads its `serial_number`, finds the matching connected camera, and builds a `CameraSource` (calibration travels inside the JSON). |
| `None` (the constructor default) | Auto-discover: `list_connected_depz_cameras()[0]` at the spec's default mode. Fine for a quick test, not what production code should rely on. |

### Calibration wire format

Single canonical shape — what the device-info endpoint returns,
what `Calibration.to_dict()` emits, what gets embedded in MKV
recordings:

```json
{
  "calibration": {
    "rectify_alpha": 0.0,
    "canvas_fit_mode": "letterbox",
    "cameras": [
      {
        "role": "left",
        "camera_index": 0,
        "model": {
          "polymorphic_name": "LENSMODEL_OPENCV8",
          "ptr_wrapper": {
            "data": {
              "parameters": {
                "f": 760.1, "ar": 1.0, "cx": 640, "cy": 360,
                "k1": ..., "k2": ..., "k3": ..., "k4": ...,
                "k5": ..., "k6": ..., "p1": ..., "p2": ...
              },
              "CameraModelCRT": { "CameraModelBase": {
                "imageSize": { "width": 1280, "height": 720 } } }
            },
            "valid": 1
          }
        },
        "transform": {
          "rotation":    { "rx": 0, "ry": 0, "rz": 0 },
          "translation": { "x":  0, "y":  0, "z":  0 }
        }
      },
      { "role": "right", "camera_index": 1,
        "model": { /* polymorphic_name omitted by cereal tag-once */ },
        "transform": { /* right cam pose in left frame */ } }
    ]
  }
}
```

Top-level is the device-info envelope (`serial_number`, `sku`,
`model_name` siblings); `Calibration.from_dict` ignores everything
outside `calibration` and dispatches purely on
`cameras[i].model.polymorphic_name`.

`OpenCVFisheyeCalibration` uses the same shape with
`polymorphic_name = "LENSMODEL_OPENCV_FISHEYE"`, four distortion
coefficients (`k1..k4`, no `p1`/`p2`/`k5`/`k6`), and SDK params
`{horizontal_fov_rad, vertical_fov_rad, canvas_fit_mode}` instead of
`rectify_alpha`.

---

## Common pitfalls

### Buffer aliasing

`Calibration.canvas_remap_pair()` returns aliased internal buffers
— the next call overwrites them. The pipeline copies what it needs
into the inflight slot. If you call this method directly, **`.copy()`
the returns** when retaining references.

`Calibration.disparity_to_depth()` and `depth_to_point_cloud()`
intentionally **do** return fresh arrays — `FramePacket`s sit in the
output queue and aliasing them would corrupt earlier frames.

### Calibration scale mismatch

If you record at 2560×720 SBS but calibration was at 1280×360 SBS,
the pipeline auto-rescales (uniform 2× / 0.5× factor). Non-uniform
mismatch (e.g. 1280×480 vs 1280×360) **fails** at `start()`.
Re-calibrate at the target resolution.

### Stream-disable refused

You can't disable a stream that something else depends on. The
pipeline won't silently let your `cloud` consumer break because
someone unset `disparity`. Disable downstream first, or pass
`cascade=True` if you really mean it.

### YUYV is bigger

YUYV at 2560×720@30 is ~110 MB/s on the wire. The recorder writes it
as rawvideo (no compression). MJPG at the same resolution is ~15
MB/s. Pick YUYV when you need bit-exact pixels (calibration capture,
low-light sensor characterization); MJPG for ordinary recording.

### Seek + paused replay

When the source is paused, `seek()` still emits exactly **one**
frame so the UI can preview the new position. After that the
pipeline goes back to idle until you resume.

---

## Class reference

The full public API — every class, method, and field in
`depz_camera_sdk.__all__` — lives in [`api.md`](api.md), generated from
the source docstrings (the single source of truth, so it never drifts
from the code). Regenerate with `make docs`.

The conceptual sections above cover *how* the pieces fit together; the
API reference covers the *exact* signatures, fields, and defaults.
