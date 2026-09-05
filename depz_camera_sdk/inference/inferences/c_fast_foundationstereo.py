"""CFastFoundationStereoInference — NVIDIA C-Fast-FoundationStereo, real-time
zero-shot stereo disparity (NVlabs, CVPR 2026).

Network: 2 inputs (left, right rectified BGR canvas), 1 output (canvas-sized
disparity in pixels). RAFT-style iterative refinement.

Six resolution variants (the ``c_ffs_<W>x<H>_opset17_sim`` ZIPs), one per
camera feed step — 640×384 (default), 1920×1088, 1280×736, 960×544,
480×288, 320×192. Picking a resolution changes the depth-map resolution
and point-cloud density; the viewer pairs it with the same-width camera
feed. The YAML sidecars are export-tooling artifacts — never downloaded:
each ONNX is static-shape and self-contained (``valid_iters=4`` and
``max_disp`` are baked into the graph; the runtime reads the input size
straight from the ONNX).

Preprocess (CONFIRMED against the NVlabs source):
- The export *strips* ImageNet normalization from the graph, so the caller must
  normalize: ``(pixel/255 - mean) / std`` with ImageNet stats, RGB order. That
  is byte-for-byte LightStereo's recipe, so the same blob params are reused.
- BGR uint8 canvas → RGB normalized float32 (ImageNet mean/std), NCHW, bundled
  into one ``cv2.dnn.blobFromImageWithParams`` call per eye.

Postprocess:
- Squeeze (1, 1, H, W) → (H, W), crop to the valid (unpadded) canvas region,
  cast float32 contiguous — identical to LightStereo.
"""


import cv2
import numpy as np

from ...calibration import CanvasLayout
from ..base import NeuralStereoInferenceBase
from ..registry import register_inference
from ..base import InferenceVariant, ModelFile


# ImageNet recipe folded into one cv2.dnn.blobFromImageWithParams call.
# blobFromImageWithParams computes (img - mean) * scale, i.e. (img - mean)/std
# with mean/std in 0-255 scale — exactly what the normalization-stripped
# C-Fast-FoundationStereo ONNX expects from the caller.
_BLOB_MEAN = (0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0, 0.0)
_BLOB_SCALE = (
    1.0 / (255.0 * 0.229),
    1.0 / (255.0 * 0.224),
    1.0 / (255.0 * 0.225),
    0.0,
)


@register_inference("c-fast-foundationstereo")
class CFastFoundationStereoInference(NeuralStereoInferenceBase):
    """C-Fast-FoundationStereo — real-time zero-shot stereo disparity.
    ImageNet-norm preprocess (the export strips in-graph normalization),
    squeeze-and-crop postprocess. The ONNX exports pin (H, W) in the graph
    and emit a single ``disparity`` output, so no INPUT_SIZE / OUTPUT_NAME
    override is needed."""

    LABEL = "C-Fast-FoundationStereo"
    ARCHITECTURE_FAMILY = "FoundationStereo"

    # (W, H) of every concrete ONNX export (the `_sim`-simplified opset17
    # builds). One per camera feed step (feed height padded up to the next
    # /32: 1920×1080→1088, 1280×720→736, 960×540→544, 640×360→384,
    # 480×270→288, 320×180→192) — the canvas letterbox adds the pad rows
    # at scale 1.0 and postprocess crops them back, so depth/point-cloud
    # come out at the true feed resolution. The 640×384 default is first —
    # it carries the family's single `preset:accurate` tag.
    _EXPORT_SIZES: tuple[tuple[int, int], ...] = (
        (640, 384),
        (1920, 1088), (1280, 736), (960, 544), (480, 288), (320, 192),
    )

    @classmethod
    def variants(cls) -> tuple[InferenceVariant, ...]:
        # ModelFile.slug is the storage key in the get-model-file-link
        # catalog; archive_member names the ONNX inside the ZIP (the _sim
        # exports name it after the slug). All resolutions share the same
        # weights + preprocess — only the baked input dims differ; the
        # per-variant valid_iters/max_disp/input_size are inert docs (read
        # by nothing at runtime; the ONNX graph is authoritative).
        out = []
        for w, h in cls._EXPORT_SIZES:
            default = (w, h) == (640, 384)
            tags = ["high-res" if w >= 960 else "low-res",
                    "weight:heavy", "year:2025"]
            if default:
                tags = ["preset:accurate", "weight:heavy", "year:2025"]
            out.append(InferenceVariant(
                slug=f"iters4-{w}x{h}",
                label=f"iters 4 / {w}×{h}",
                files_to_download=(
                    ModelFile(
                        slug=f"c_ffs_{w}x{h}_opset17_sim",
                        filename="model.onnx",
                        archive_member=f"c_ffs_{w}x{h}_opset17_sim.onnx"),
                ),
                configurations={
                    "valid_iters": 4,
                    "max_disp": 320,
                    "input_size": (h, w),
                    "selector_meta": {
                        "task": "stereo",
                        "speed_tier": 3,
                        "accuracy_tier": 4,
                        "min_engine_class": "nvidia",
                        "recommended_engine_class": "nvidia",
                        "size_mb": 0.0,
                        "params_m": 0.0,
                        "published_fps": {"nvidia": 71},
                        "accuracy_metric": "Zero-shot, ~FoundationStereo quality (4 iters)",
                        "summary": "Top accuracy on an NVIDIA GPU.",
                        "display_family": "C-Fast-FoundationStereo",
                        "tags": tags,
                        "neural": True,
                        "wizard_visible": True,
                        "license": "NVIDIA Open Model Agreement",
                        "license_url": "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/",
                    },
                },
            ))
        return tuple(out)

    def _init_extra_state(self) -> None:
        self._blob_params = cv2.dnn.Image2BlobParams(
            scalefactor=_BLOB_SCALE,
            size=(0, 0),
            mean=_BLOB_MEAN,
            swapRB=True,
            ddepth=cv2.CV_32F,
        )

    def preprocess(self,
                   canvas_l: np.ndarray, canvas_r: np.ndarray,
                   ) -> tuple[np.ndarray, np.ndarray]:
        L = cv2.dnn.blobFromImageWithParams(canvas_l, self._blob_params)
        R = cv2.dnn.blobFromImageWithParams(canvas_r, self._blob_params)
        return L, R

    def postprocess(self,
                    raw_output: np.ndarray,
                    canvas_layout: CanvasLayout,
                    ) -> np.ndarray:
        arr = np.asarray(raw_output)
        disp = np.squeeze(arr)
        if disp.ndim != 2:
            raise RuntimeError(
                f"model output shape {tuple(arr.shape)} does not yield "
                f"a 2D map after squeeze (got {tuple(disp.shape)})")
        if disp.dtype != np.float32:
            disp = disp.astype(np.float32, copy=False)
        y, x = canvas_layout.pad_y, canvas_layout.pad_x
        h, w = canvas_layout.valid_h, canvas_layout.valid_w
        return np.ascontiguousarray(disp[y: y + h, x: x + w])
