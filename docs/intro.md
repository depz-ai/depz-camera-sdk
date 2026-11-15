# Python SDK

`depz-camera-sdk` turns a stereo (side-by-side) or mono DEPZ camera — or a
previously-recorded MKV file — into a stream of frame-numbered packets carrying
raw eyes, rectified eyes, disparity, metric depth, and a 3D point cloud.

- **Cross-platform** — Windows 10+ x64, Windows 11 ARM64, Linux x86_64 and
  ARM64 (aarch64, glibc 2.28+), and macOS 14+ on Apple Silicon, shipped as
  pre-built wheels for Python 3.10 or newer. On ARM64 boards: NVIDIA Jetson (JetPack 6 and 7, GPU through
  TensorRT) and Rockchip (RK3588, RK3576, RK3568, RK3566, RK3562, …, NPU
  through RKNN2).
- **Neural & classical stereo** — Lite Any Stereo,
  C-Fast-FoundationStereo, LightStereo and CREStereo ONNX nets, OpenCV SGBM /
  StereoBM block matchers, plus monocular depth (Depth Anything V2,
  Metric3D V2).
- **Your hardware, your backend** — OpenVINO and ONNX Runtime engines with CPU,
  CUDA, DirectML, CoreML, MIGraphX and QNN execution, TensorRT on NVIDIA
  desktop GPUs and Jetson, and RKNN2 on Rockchip NPUs; the model catalog downloads and caches weights on
  first use.
- **Record & replay** — capture the raw camera stream to MKV and re-run the
  exact same frames through any pipeline configuration later.

## Install

Needs Python 3.10 or newer. Install into a virtual environment — current
Debian / Ubuntu and Homebrew Pythons refuse a system-wide `pip install`:

```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: python -m venv .venv; .venv\Scripts\Activate.ps1
pip install "depz-camera-sdk[openvino]"   # Windows / Linux x86_64 default (Intel & AMD CPU, Intel iGPU/NPU)
pip install "depz-camera-sdk[cpu]"        # macOS (CoreML on Apple Silicon) and Linux ARM64
pip install "depz-camera-sdk[cpu,rknn]"   # Rockchip boards — adds the NPU (RKNN2)
```

On Windows ARM64 use a native ARM64 Python and `[cpu]`, or `[qnn]` for a
Qualcomm NPU. PyPI has no OpenCV wheel for that platform, so pip needs one
extra argument — see the [guide](guide.md#installation).

On an NVIDIA Jetson, create the environment with
`python3 -m venv --system-site-packages .venv` and install
`"depz-camera-sdk[cpu]"`: TensorRT is part of JetPack and the SDK uses it
directly. Both board families have their own section in the guide.

A bare `pip install depz-camera-sdk` is "pure" — no inference engines at
all, so it cannot run a neural model until you add an engine. Add an ONNX
Runtime extra for a GPU: `cuda12-bundled` (NVIDIA, CUDA 12, self-contained),
`cuda12` (NVIDIA, you provide the CUDA 12 runtime), `cuda13-bundled` /
`cuda13` (the same on CUDA 13 — newer GPUs and drivers, Python 3.11+), `dml`
(Windows, any DX12 GPU) or `migraphx` (Linux AMD, system ROCm). The ONNX
Runtime extras are mutually exclusive — pick at most one; `openvino`
combines with any of them, e.g. `[openvino,cuda12-bundled]`. For NVIDIA
TensorRT (three to four times less GPU time per frame) add `tensorrt12`
next to a `cuda12` extra or `tensorrt13` next to a `cuda13` one, e.g.
`[openvino,cuda12-bundled,tensorrt12]` — a multi-GB download, Python
3.10–3.13. See all options
in the [install selector](/developers/sdk) and the
[guide](guide.md#installation).

Check the install — this prints the engines the SDK can use (an empty list
means no engine extra is installed):

```bash
python -c "import depz_camera_sdk as d; print(d.available_engines())"
```

Just want the viewer app, no Python? One command installs it into a private
environment ([more platforms & details](guide.md#installation)):

```sh
# macOS 14+ (Apple Silicon)
curl -fsSL https://github.com/depz-ai/depz-camera-sdk/releases/latest/download/Install-CameraViewer-macOS.sh | sh -s -- --add-to-path
```

## Quickstart

Plug in a DEPZ camera and run:

```python
from depz_camera_sdk import DepthPipeline

for frame in DepthPipeline():
    print(f"frame {frame.frame_number}, depth shape {frame.depth.shape}")
```

That's the whole hello-world: `DepthPipeline()` finds the first connected DEPZ
camera, loads its default stereo model (Lite Any Stereo V2 S at 640×384)
and starts yielding frames. The model is fixed; only the inference engine is
picked automatically, from the engines you installed — so the default needs
an engine extra such as `[openvino]` or `[cpu]`.

## A fuller example

Everything the one-liner guessed can be made explicit — pick the camera, the
capture mode, the model and the inference engine yourself:

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

## Where to next

- **[Guide](guide.md)** — the hand-written walkthrough: recording and replay,
  the model catalog, engines, calibration, camera controls, filters, and
  common pitfalls.
- **[API Reference](api.md)** — every public class and function, generated
  from the SDK's docstrings.
- **Docs for LLMs** — this documentation as one raw-Markdown file:
  [/llms-full.txt](/llms-full.txt).
