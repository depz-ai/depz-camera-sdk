# DEPZ Camera SDK

Python SDK and browser viewer for [DEPZ](https://depz.ai) stereo depth
cameras — capture, MKV record/replay, rectification, and depth inference
(OpenVINO / ONNX Runtime engines: CPU, Intel iGPU/NPU, NVIDIA CUDA,
DirectML, CoreML).

- **Docs:** [depz.ai/developers/docs/depth-cameras](https://depz.ai/developers/docs/depth-cameras)
  · for LLMs/agents: [depz.ai/llms-full.txt](https://depz.ai/llms-full.txt)
- **Releases:** [wheels + one-command installers](https://github.com/depz-ai/depz-camera-sdk/releases/latest)
- **Issues:** bug reports and questions are welcome right here.

## Install

Viewer, no Python knowledge needed (installs into a private environment):

```powershell
# Windows x64 (normal PowerShell window, not "Run as Administrator")
irm https://github.com/depz-ai/depz-camera-sdk/releases/latest/download/Install-CameraViewer-Windows.ps1 | iex
```

```sh
# Linux x86_64
curl -fsSL https://github.com/depz-ai/depz-camera-sdk/releases/latest/download/Install-CameraViewer-Ubuntu-x86_64.sh | sh
# macOS 11+ (Intel + Apple Silicon)
curl -fsSL https://github.com/depz-ai/depz-camera-sdk/releases/latest/download/Install-CameraViewer-macOS.sh | sh
```

Developers — from PyPI (pick extras for your hardware):

```bash
pip install depz-camera-sdk[openvino]              # SDK
pip install depz-camera-viewer[openvino]           # SDK + browser viewer
```

## What's in this repository

| Path | Contents |
|---|---|
| [Releases](https://github.com/depz-ai/depz-camera-sdk/releases) | Wheels for Linux/Windows/macOS, end-user installers, release manifest |
| `depz_camera_sdk/` | Read-only mirror of the SDK's open source — the extension surface you subclass and read: control descriptors, filter framework, inference/engine ABCs + registries, data types, package API |
| `docs/` | The SDK guide and generated API reference |

The SDK ships as compiled wheels: modules carrying proprietary
implementation (camera drivers, calibration math, model wrappers, the
pipeline core) are Cython-compiled and are **not** mirrored here. The
mirrored `.py` files are exactly the ones that ship as plain source inside
every wheel — updated automatically on each release.

This mirror is for reading and reference; it is not an installable
package. Install from PyPI or the releases above.

## License

Dual-licensed:

- **Contents of this repository** (the mirrored `.py` sources and docs) —
  [MIT](LICENSE). Read them, copy them into your code, subclass freely.
- **The distributed Software** (wheels on PyPI and in the releases,
  including the compiled modules, and the viewer) —
  [DEPZ Camera SDK License Agreement](EULA.txt): free to use with DEPZ
  cameras and to embed in your own products; no standalone redistribution
  or reverse engineering of the compiled modules.
