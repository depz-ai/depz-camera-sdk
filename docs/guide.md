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
  - [NVIDIA Jetson (TensorRT)](#nvidia-jetson-tensorrt)
  - [Rockchip NPU (RKNN)](#rockchip-npu-rknn)
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
   - `LiteAnyStereoInference` / `LiteAnyStereoMInference` (the default
     model is Lite Any Stereo V2 S), `DepzStereoInference`,
     `LightStereoInference`, `CREStereoInitialInference`,
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

The SDK ships on PyPI as `depz-camera-sdk` for Windows 10+ x64, Windows 11
ARM64, Linux x86_64 and ARM64 (aarch64) with glibc 2.28+, and macOS 14+ on
Apple Silicon, for any Python ≥ 3.10 (stable-ABI wheels). A bare install is
"pure" — no inference engines at all;
add extras for your hardware. `[openvino]` covers Intel CPU/iGPU/NPU and
AMD CPU (Windows/Linux x86_64 only). The ONNX Runtime extras are mutually
exclusive — pick at most one:

```bash
pip install "depz-camera-sdk"                     # pure: no engines
pip install "depz-camera-sdk[openvino]"           # the x86_64 default
pip install "depz-camera-sdk[openvino,cuda12-bundled]" # + NVIDIA GPU, CUDA 12, self-contained (~1.5 GB)
pip install "depz-camera-sdk[openvino,cuda12]"    # + NVIDIA GPU, you provide the CUDA 12 runtime
pip install "depz-camera-sdk[openvino,cuda13-bundled]" # + NVIDIA GPU, CUDA 13, self-contained (~1.5 GB)
pip install "depz-camera-sdk[openvino,cuda13]"    # + NVIDIA GPU, you provide the CUDA 13 runtime
pip install "depz-camera-sdk[openvino,dml]"       # + AMD GPU on Windows (any DX12 GPU)
pip install "depz-camera-sdk[openvino,migraphx]"  # + AMD GPU on Linux (system ROCm + MIGraphX)
pip install "depz-camera-sdk[cpu]"                # Linux ARM64 / macOS / Windows ARM64 baseline
pip install "depz-camera-sdk[qnn]"                # Windows ARM64 with a Qualcomm NPU (instead of [cpu])
pip install "depz-camera-sdk[cpu,rknn]"           # Rockchip boards: + the NPU (RKNN2 runtime)
pip install "depz-camera-sdk[openvino,cuda12-bundled,tensorrt]"  # x86_64: + NVIDIA TensorRT, used directly
```

(The quotes matter in zsh, the default macOS shell, where bare `[...]`
is a glob pattern.)

NVIDIA has two CUDA lines; an environment is on exactly one of them.
`cuda12*` (ONNX Runtime 1.19–1.26) runs on every supported GPU — GTX
10-series and newer, driver 525.60+ on Linux / 527.41+ on Windows.
`cuda13*` (ONNX Runtime 1.27+) needs a GTX 16-series / RTX 20-series or
newer GPU, driver 580+ and Python 3.11+. `-bundled` ships the CUDA + cuDNN
libraries (~1.5 GB); without it you provide that CUDA's Toolkit and cuDNN 9.
The SDK reads the line off the installed ONNX Runtime — there is nothing to
configure. To switch lines, recreate the environment.

Two ARM64 board families have their own accelerator and their own
section below: [NVIDIA Jetson](#nvidia-jetson-tensorrt) (JetPack 6 and 7,
GPU through TensorRT — `[cpu]` in an environment that can see JetPack's
packages) and [Rockchip](#rockchip-npu-rknn) (NPU through RKNN2 —
`[cpu,rknn]`).

On macOS there is no CUDA/DirectML/OpenVINO: `[cpu]` installs the stock
ONNX Runtime wheel, which on Apple Silicon includes the CoreML EP
(GPU/ANE) — and everything else in this guide (live capture, recording,
replay, all inferences) works the same as on Linux and Windows.

On Windows ARM64, PyPI has no OpenCV wheel, so the SDK depends on
`opencv-python-headless` there and every GitHub release carries a
`win_arm64` build of it. Tell pip where to find it (`X.Y.Z` = the version
you install; use a native ARM64 Python):

```powershell
pip install "depz-camera-sdk[cpu]==X.Y.Z" --find-links https://github.com/depz-ai/depz-camera-sdk/releases/expanded_assets/vX.Y.Z
```

If you want the browser viewer rather than the raw SDK, either
`pip install "depz-camera-viewer[cpu]"` (same extras) or use the no-Python
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
- ② **`FrameSource.decode_to_bgr()` + `Calibration.canvas_remap()`**
  — decodes wire bytes to BGR, splits SBS halves (stereo) or passes
  through (mono), then rectifies onto the inference canvas in one
  `cv2.remap` per eye.
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

Every setter returns once the worker has applied the change, and says
so when it could not: if the worker does not pick the command up in
time (2 s for cheap controls, 10 s for calibration rebuilds and seeks)
the command is withdrawn and `TimeoutError` is raised — it will not be
applied later behind your back. On a pipeline that is not running the
setters apply directly, so you can configure filters, streams and
calibration controls before `start()`; `seek` / `step_replay` need a
running pipeline and raise `RuntimeError` otherwise.

`start()` either succeeds or leaves nothing behind (source closed,
engine shut down). If the worker dies — a camera that stops delivering
makes `grab()` raise — `is_running` turns False and `last_error` holds
the reason; call `stop()` to release the device. A pipeline is
single-shot: after `stop()` build a new one.

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

With no arguments the pipeline takes the first connected DEPZ camera
and `DepthPipeline.DEFAULT_INFERENCE` — Lite Any Stereo V2 S at 640×384
(MIT-licensed, runs on a CPU). The model is downloaded on first use;
that happens inside the constructor, so build the inference yourself
(next section) if you want to show progress or choose the engine.

### Five-minute, fully-explicit form

```python
from depz_camera_sdk import (
    DepthPipeline, CameraSource, list_connected_depz_cameras,
)
from depz_camera_sdk.inference import build_inference

camera_info  = list_connected_depz_cameras()[0]
capture_mode = camera_info.spec.get_mode("1280x720@30/MJPG/stereo")
camera_source = CameraSource(camera_info, capture_mode)

inference = build_inference("lite-any-stereo", "v2-s-640x384",
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
from `source.camera_info` (live) or the input MKV's attachment (when
re-recording from replay). No sidecar files — the calibration and its
history and the camera-state snapshots all travel inside the MKV.

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

Internally runs a **pump thread** that reads from V4L2 / DirectShow /
AVFoundation at full rate and hands the latest frame to the pipeline
(drop-to-latest — older frames inference didn't pick up are
discarded). `grab()` blocks on a condition variable, so there's no
wasteful polling; it returns `None` on a timeout and raises
`RuntimeError` once the pump thread has died (device unplugged, stream
ended), so a dead camera is an error rather than an endless silence.

`CameraSource(cam, output_eye=(w, h))` delivers frames downscaled to
that per-eye size while capturing at the mode's size;
`src.capture_size` is the captured `(width, height)` — the size
recordings are made at.

The bound `DepzCameraInfo` lets `DepthPipeline.start_recording`
derive the `camera_info.json` MKV attachment automatically.

#### `MkvSource` — replay

```python
from depz_camera_sdk import MkvSource

src = MkvSource("/path/to/recording.mkv",
                replay_mode="realtime",    # "every_frame" = block on consumer
                loop=False,
                build_frame_index=True)    # default; needed for step API
src.open()
src.frame_count                            # exact, from the frame index
src.loop = True                            # can be switched while playing
src.embedded_camera_info                   # the camera_info.json payload
src.pause(); src.resume()
src.seek_relative_us(5_000_000)            # 5 s into the file
src.step_one_forward()                     # exact ±1 frame (paused only)
```

The pipeline mirrors the same API — `pipeline.pause()` / `resume()` /
`is_pausable` / `is_paused` / `step_replay("forward"|"backward")` —
so application code rarely needs to touch the source directly.

**End of file.** Without `loop=True` a replay is finite: `for frame in
pipeline:` ends after the last recorded frame, and a blocking
`pipeline.get()` returns `None`. `pipeline.at_eof` tells "finished"
apart from "nothing yet" when you poll with a timeout. The pipeline
itself keeps running at EOF, so you can seek back and continue:

```python
pipeline = DepthPipeline(source=MkvSource(path, replay_mode="every_frame"),
                         inference=inference)
with pipeline:
    while (frame := pipeline.get()) is not None:   # every frame, once
        ...
    pipeline.seek(fraction=0.0)                    # rewind — frames flow again
```

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
- downloading weights into a local cache — with a socket timeout, the
  byte count checked against what the server announced, and the
  declared size / sha256 verified where the catalog has them,
- atomic writes (no half-downloaded files on Ctrl-C) and a per-file
  lock, so several processes can ask for the same model at once,
- handing back a ready-to-use `DepthInferenceBase` for the pipeline.

**No models are bundled in the wheel.** First use fetches on demand.

#### Where models live on disk

Models cache under `<cache>/models/<inference>/<variant>/<file>`, where
`<cache>` is the per-user cache directory:

| OS | Default cache root |
| --- | --- |
| Linux | `$XDG_CACHE_HOME/depz`, else `~/.cache/depz` |
| macOS | `~/Library/Caches/depz` |
| Windows | `%LOCALAPPDATA%\depz\Cache` |

It does not depend on the working directory. To keep everything
somewhere else (a data volume, a read-only image), redirect it once at
startup — before building any inference:

```python
from depz_camera_sdk import get_cache_dir, set_cache_dir

set_cache_dir("/data/depz-cache")     # ~ and $VARS are expanded
print(get_cache_dir())                # always an absolute path
```

Every component (model downloader, engine caches, the viewer's
camera-info store) writes under that root. Next to each downloaded
file sits `<file>.meta.json` (size, sha256, source slug) — the cache
check uses it to notice a damaged file — and `<file>.lock`.

#### Walking the catalog

```python
from depz_camera_sdk.inference import available_inferences

for cls in available_inferences():
    domain = "depth" if cls.OUTPUTS_DEPTH else "disparity"
    print(f"{cls.SLUG} — {cls.LABEL} ({domain})")
    for v in cls.variants():
        mark = "✓ cached" if v.is_cached else "  needs download"
        h, w = v.input_size
        print(f"   {v.slug:<32} {v.label:<32} {w}×{h}   {mark}")
```

Each `InferenceVariant` carries `slug`, `label`, `input_size` (H, W),
the `files_to_download`, and `configurations["selector_meta"]` — the
facts the viewer's model picker shows (speed / accuracy tier, license,
measured disparity range, …).

Typical printout (truncated — CREStereo Combined alone has 20 variants):

```
c-fast-foundationstereo — C-Fast-FoundationStereo (disparity)
   iters4-640x384                   iters 4 / 640×384                  needs download
   iters4-1920x1088                 iters 4 / 1920×1088                needs download
   ...
crestereo-combined — CREStereo Combined (disparity)
   iter2-240x320                    iter 2 / 240×320                   needs download
   iter2-320x480                    iter 2 / 320×480                   needs download
   ...
crestereo-initial — CREStereo Initial (disparity)
   iter2-160x240                    iter 2 / 160×240                   needs download
   ...
depth-anything-v2-small — Depth Anything V2 Small (depth)
   518x518                          518×518 (default)                  needs download
depz-stereo — DEPZ Stereo (disparity)
   depz-stereo-320x192              320×192                            needs download
   depz-stereo-1920x1088            1920×1088                          needs download
   ...
lightstereo — LightStereo (disparity)
   s-sceneflow-opt-256x512          S / SceneFlow / 256×512 (opt)      needs download
   s-sceneflow-opt-576x960          S / SceneFlow / 576×960 (opt)      needs download
   s-sceneflow-origin-256x512       S / SceneFlow / 256×512 (origin)   needs download
lite-any-stereo — Lite Any Stereo (disparity)
   v2-s-320x192                     V2 S / 320×192                     needs download
   v2-s-640x384                     V2 S / 640×384                   ✓ cached
   ...
lite-any-stereo-m — Lite Any Stereo M (disparity)
   v2-m-640x384                     V2 M / 640×384                     needs download
   v2-m-1920x1088                   V2 M / 1920×1088                   needs download
   ...
metric3d-v2-vit-small — Metric3D V2 (ViT-Small) (depth)
   vit-small-504x504                ViT-Small / 504×504                needs download
rectify-only — Rectify only (no inference) (disparity)
   256x512                          256×512 (fast)                   ✓ cached  (no download)
   ...
sgbm — SGBM (classical, CPU) (disparity)
   256x512                          256×512 (fast)                   ✓ cached  (no download)
   360x640                          360×640 (balanced)               ✓ cached  (no download)
   480x720                          480×720 (detailed)               ✓ cached  (no download)
stereo-bm — StereoBM (classical, CPU, fastest) (disparity)
   256x512                          256×512 (fast)                   ✓ cached  (no download)
   ...
```

Licenses differ per model — `selector_meta["license"]` says which.
Lite Any Stereo (MIT) is the default; LightStereo's upstream is
academic-use-only.

#### Building an inference from a catalog variant

The one-liner. `build_inference` walks the steps below internally —
this is what the hello-world uses.

```python
from depz_camera_sdk.inference import build_inference

inference = build_inference(
    "lite-any-stereo", "v2-s-640x384",
    download_progress_cb=lambda done, total: print(f"\r{done}/{total}", end=""),
)
```

Downloads are silent unless you pass `download_progress_cb(done_bytes,
total_bytes)`. For a ready-made console progress line use
`download_progress_cb=stderr_progress()` (from `depz_camera_sdk`).

If you want to do it by hand (testing one stage in isolation):

```python
from depz_camera_sdk import recommend_engine
from depz_camera_sdk.engines import make_engine
from depz_camera_sdk.inference import get_inference

cls     = get_inference("lite-any-stereo")
variant = cls.get_variant("v2-s-640x384")

# 1. Ensure cached + return the absolute paths of every declared
#    file. files[0] is the entry-point (model.onnx); siblings
#    (model.onnx_data for ONNX with external data) follow in
#    declaration order.
files = variant.resolve_files()

# 2. Build the engine around the entry-point file. This is where
#    engine options go (cache_dir=..., nireq=..., see "Inference
#    engines" below). A network whose export has dynamic dimensions
#    declares INPUT_SIZE on its class and needs
#    input_shape=(1, 3, *cls.INPUT_SIZE) here — build_inference does
#    that for you.
engine = make_engine(recommend_engine(), str(files[0]))

# 3. Construct the inference. ``inference_kwargs`` on the variant
#    carry per-variant ctor args (empty for neural networks;
#    canvas_size for SGBM/StereoBM/RectifyOnly).
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

Call `set_cache_dir(...)` first to fill a directory you then ship as a
read-only image / squashfs / NFS share. (A read-only cache serves
models fine; the engines' own compile caches are simply skipped when
they cannot be written.)

### Inference engines

Four backends with the same async interface, identified by **slug
strings** — `<backend>-<device>`:

| Backend | Slugs | Hardware |
|---|---|---|
| OpenVINO | `openvino-CPU`, `openvino-GPU`, `openvino-NPU` | Intel CPU / GPU / NPU, AMD CPU |
| ONNX Runtime | `onnx-cuda`, `onnx-dml`, `onnx-migraphx`, `onnx-coreml`, `onnx-qnn`, `onnx-cpu` | NVIDIA / AMD / Apple / Qualcomm, any CPU |
| TensorRT (used directly) | `tensorrt-fp16` (`tensorrt-fp32` on request) | NVIDIA Jetson; NVIDIA desktop GPUs with the `[tensorrt]` extra |
| RKNN2 | `rknn-npu` | Rockchip NPU |

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

**Engine options** are keyword arguments of `make_engine` (to use them
with a catalog model, build the engine yourself as shown in "Building
an inference from a catalog variant"). `make_engine` rejects an option
the chosen engine kind does not have, naming it:

- `cache_dir` (both kinds) — override the SDK-wide cache root
  (`get_cache_dir()`) for this engine's on-disk cache:
  `<cache_dir>/openvino/` compiled blobs, `<cache_dir>/onnx-opt/`
  optimized graphs.
- `input_shape` (both) — pin a `(1, 3, H, W)` shape on a model exported
  with dynamic dimensions; required for such models (Metric3D V2,
  Depth Anything V2 — their classes declare `INPUT_SIZE` and
  `build_inference` passes it).
- `output_name` (both) — which output to read on a multi-head model.
- `do_models_cache` (OpenVINO) — set False to disable the compiled-
  model cache entirely (read-only filesystems / one-shot scripts).
- `nireq` (OpenVINO) — async-queue depth.
- `num_threads` (ONNX Runtime) — how many `session.run` calls may be in
  flight at once (default 1).

OpenVINO uses `AsyncInferQueue`; ONNX Runtime uses a small thread
pool plus `session.run()`. Either way, the pipeline never blocks on
inference — it submits and pulls completed futures FIFO. First GPU
compile is ~30–60 s; subsequent loads are sub-second thanks to the
on-disk cache.

### NVIDIA Jetson (TensorRT)

Supported: **JetPack 6** (Ubuntu 22.04, Python 3.10, TensorRT 10.3) and
**JetPack 7** (Ubuntu 24.04, Python 3.12, TensorRT 10.x) — the Orin
family, and Thor on JetPack 7. The GPU is driven through TensorRT
directly, in half precision (FP16); the engine slug is `tensorrt-fp16`.
No ONNX Runtime GPU build is involved.

TensorRT's Python package is part of JetPack and belongs to the
**system** Python, so the environment must be created from it and be
allowed to see system packages:

```bash
sudo apt install python3-libnvinfer        # already there on a full JetPack install
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install "depz-camera-sdk[cpu]"
python -c "import depz_camera_sdk as d; print(d.available_engines())"
# ['tensorrt-fp16', 'onnx-cpu']
```

(The viewer's one-line installer does the same on its own when it finds
a Jetson.) `[cpu]` adds ONNX Runtime for the CPU device; packages
installed into the environment — numpy, OpenCV — take precedence over
JetPack's older copies.

What to expect:

- **The first start of each model takes minutes.** TensorRT compiles the
  ONNX file into a *plan* for this exact GPU and TensorRT version and
  stores it in `<cache_dir>/tensorrt/`. Later starts load the plan in
  about a second. A plan is not portable: after a JetPack upgrade, or on
  another module, it is rebuilt automatically.
- **Precision.** Almost every layer runs in FP16. Layers around
  constants half precision cannot represent (an epsilon of `1e-12` in a
  normalisation becomes zero) are kept in FP32 automatically, and a
  freshly built plan is checked for NaN / infinity; a model that still
  fails that check is rebuilt in FP32. `tensorrt-fp32` forces full
  precision for one model: `make_engine("tensorrt-fp32", path)`.
- The same engine works on x86_64 desktops with
  `pip install "depz-camera-sdk[openvino,cuda12-bundled,tensorrt]"`. There
  it ranks below `onnx-cuda` (which starts instantly), so pick it
  explicitly.

### Rockchip NPU (RKNN)

Supported: Rockchip processors served by the **RKNN2** runtime —
RK3588, RK3588S, RK3576, RK3568, RK3566, RK3562, RV1126B, … (Rockchip
extends the list with every release; the older RK1808 / RK3399Pro /
RV1109 / RV1126 use the previous-generation RKNN and are not
supported). The engine slug is `rknn-npu`.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install "depz-camera-sdk[cpu,rknn]"     # Python 3.10 – 3.12
python -c "import depz_camera_sdk as d; print(d.available_engines())"
# ['rknn-npu', 'onnx-cpu']
```

The board's Linux image must provide the NPU kernel driver and
`/usr/lib/librknnrt.so` (RKNN runtime 2.3 or newer; images from the
board vendors include both).

An NPU does not run ONNX files. Each model is converted once into an
`.rknn` file **for one processor family** — an RK3576 file does not load
on an RK3588. Models are converted without quantization, so the NPU
computes in FP16 and no calibration images are needed. The converter is
Rockchip's `rknn-toolkit2`, which pins old numpy / torch and therefore
lives in its own environment (on the board itself or on any x86_64
Linux PC):

```bash
# 1. download the model with the SDK (any engine, e.g. on the board)
python -c "from depz_camera_sdk import set_cache_dir; \
from depz_camera_sdk.inference import get_inference; \
set_cache_dir('~/.cache/depz'); \
print(get_inference('lite-any-stereo').get_variant('v2-s-640x384').resolve_files())"

# 2. convert it, in a separate environment
uv venv --python 3.10 ~/rknn-convert
uv pip install --python ~/rknn-convert/bin/python rknn-toolkit2==2.5.0 "setuptools<80"
~/rknn-convert/bin/python tools/model_export/convert_rknn.py \
    --models-dir ~/.cache/depz/models --platform rk3576
```

The converter writes `model.<platform>.fp16.rknn` and a small `.json`
description next to each `model.onnx` — exactly where the engine looks.
`tools/model_export/convert_rknn.py` is a single file with no SDK
imports; copy it anywhere. Starting a model that has not been converted
fails with the command to run.

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
calib.control_set("rectify_alpha", 0.0)
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
| `CameraSource` | `source.camera_info.to_dict()` — server-shape, identical to the device-info endpoint response. |
| `MkvSource` (re-recording from replay) | The `camera_info.json` attachment of the input MKV, embedded verbatim. |
| Custom source / mock | Reconstructed from `pipeline.active_calibration.to_dict()` plus identity off `pipeline.camera_info`. |

A live camera fans out **every** captured frame to the recorder via
the source's pump thread (so you don't lose frames if inference
can't keep up); so does a `realtime` replay. An `every_frame` replay
has no pump thread and is recorded at worker rate — every frame the
pipeline consumes, which in that mode is every frame.

The file is recorded at the size the camera **captures**
(`source.capture_size`), also when the source delivers downscaled
frames (`CameraSource(output_eye=...)`).

If the recorder's writer fails (disk full, …) `recording_status.error`
says so and `stop_recording()` still returns — its
`RecordingResult.error` carries the reason and the partial file is
left at `result.path`.

`pipe.capture_single_frame("/tmp/still.mkv")` saves just the next raw
frame as a one-frame MKV — a still that keeps its calibration and can
be replayed through the pipeline like any recording.

YUYV captures are written as `rawvideo` (FOURCC tag `YUY2`). MJPG
is remuxed as-is. Camera-state snapshots (one per second) and
calibration updates are muxed live into two subtitle tracks
(`camera_state`, `calibration`) alongside the video, so the file is
written in a single pass — no re-mux, no external `ffmpeg`.

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
`NeuralStereoInferenceBase` (two eyes in, **disparity** out) or
`NeuralMonoInferenceBase` (left eye in, **depth** out; inherits the
`depth_scale` / `depth_offset` affine controls). The bases already do
the usual pre- and post-processing, driven by class attributes:

- `BLOB_MEAN` / `BLOB_SCALE` — input normalisation,
  `(RGB - mean) * scale` per channel. The default feeds raw RGB in
  [0, 255] (for exports that normalise inside the graph); set both to
  `IMAGENET_BLOB_MEAN` / `IMAGENET_BLOB_SCALE` for ImageNet-normalised
  input.
- `OUTPUT_CHANNEL` — for a `(1, C, H, W)` output, which channel is the
  map (default `None`: the output squeezes to one 2-D map).
- `INPUT_SIZE` — `(H, W)` to pin an export with dynamic dimensions.
- `OUTPUT_NAME` — which output to read on a multi-head model.

So a network with a standard interface is a class with no methods:

```python
from depz_camera_sdk import DepthPipeline, recommend_engine
from depz_camera_sdk.engines import make_engine
from depz_camera_sdk.inference import (
    IMAGENET_BLOB_MEAN, IMAGENET_BLOB_SCALE, NeuralStereoInferenceBase,
)


class MyStereoNet(NeuralStereoInferenceBase):
    SLUG = "my-stereo-net"
    BLOB_MEAN = IMAGENET_BLOB_MEAN
    BLOB_SCALE = IMAGENET_BLOB_SCALE


inf = MyStereoNet(engine=make_engine(recommend_engine(),
                                     "/abs/path/to/my-model.onnx"))

with DepthPipeline(source=src, inference=inf, enable_cloud=True) as pipe:
    pkt = pipe.get(timeout=10.0)
    print(pkt.depth.shape, pkt.point_cloud.shape)
```

The pipeline takes the instance as-is — rectification onto the
model's input canvas, disparity → depth → point-cloud projection,
filters, recording, and the web-viewer wiring all come for free.
Override a method only where the network really differs:

- `preprocess(canvas_l, canvas_r)` — BGR uint8 canvases → tuple of
  NCHW float32 blobs, in the order the model declares its inputs
  (`self._to_blob(canvas)` applies the class recipe to one canvas).
  Mono networks receive `canvas_r=None`.
- `postprocess(raw_output, canvas_layout)` — raw engine output → 2-D
  float32 map of the canvas's valid region
  (`self._extract_map(raw_output, canvas_layout)` does the squeeze /
  channel pick / cast / crop). Stereo returns disparity in canvas
  pixels; mono returns depth.

Registering in the catalog is only needed if you also want the model
listed / downloadable via `build_inference`: decorate the class with
`@register_inference("slug")` (which sets `SLUG`) and declare its
variants in a `_variants()` classmethod — `onnx_variant(...)` builds
the usual single-file entry. A direct instance like above skips that
entirely.

### Custom sources

Both built-ins extend `FrameSource`. To plug in your own (e.g. a
network stream): inherit `FrameSource`, implement
`open()` / `grab()` / `close()` plus the relevant capability
properties. Defaults already cover most things — sane no-ops for
`add_fanout`, `camera_controls`, `seek_relative_us`, etc. If your bytes
are MJPG or YUYV, the default `decode_to_bgr` works. `grab()` returns
`None` when there is nothing right now and raises when the source can
never deliver again. A source with its own capture thread returns
`True` from `add_fanout(cb)` and calls `cb` for every frame; otherwise
the pipeline feeds the recorder itself.

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

Subclass `Calibration`, set `POLYMORPHIC_NAME`, declare the lens
model's runtime controls as typed-control class attributes, implement
`_extract_distortion`, `_distortion_params_dict` and
`_compute_rectification`, then decorate with `@register_calibration`. The base does the rest. Use
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
| `*.mkv` path (any case; `/` or `\` separators) | Replay via `MkvSource`. |
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

`Calibration.canvas_remap()` returns aliased internal buffers
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
