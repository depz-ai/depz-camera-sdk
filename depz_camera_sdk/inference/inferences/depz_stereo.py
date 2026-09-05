"""DepzStereoInference — DEPZ's own fast stereo disparity network.

Network: 2 inputs (left, right BGR canvas), 1 output (canvas-sized
disparity in pixels). Preprocess/postprocess are an exact copy of
LightStereo's recipe (same ImageNet-norm blob params, same
squeeze-and-crop postprocess) — only the weights and the resolution
lineup differ.

Preprocess:
- BGR uint8 → RGB normalized float32 (ImageNet mean/std), NCHW.
- Bundled into one ``cv2.dnn.blobFromImageWithParams`` call per eye.

Postprocess:
- Squeeze (1, 1, H, W) → (H, W).
- Crop to the valid region (``canvas_layout.valid_w × valid_h`` at
  ``pad_x, pad_y``) — strips letterbox padding.
- Cast to float32 contiguous.
"""

import cv2
import numpy as np

from ...calibration import CanvasLayout
from ..base import InferenceVariant, ModelFile, NeuralStereoInferenceBase
from ..registry import register_inference

# ImageNet recipe folded into one cv2.dnn.blobFromImageWithParams call —
# identical to LightStereo's.
_BLOB_MEAN = (0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0, 0.0)
_BLOB_SCALE = (
    1.0 / (255.0 * 0.229),
    1.0 / (255.0 * 0.224),
    1.0 / (255.0 * 0.225),
    0.0,
)


@register_inference("depz-stereo")
class DepzStereoInference(NeuralStereoInferenceBase):
    """DEPZ Stereo — fast disparity stereo, DEPZ's own model. ImageNet-
    norm preprocess, squeeze-and-crop postprocess. ONNX exports pin
    (H, W) in the graph, so no INPUT_SIZE override needed."""

    LABEL = "DEPZ Stereo"

    # (W, H) of every concrete ONNX export (the `_sim`-simplified opset20
    # builds). Each camera feed step has a SAME-WIDTH export whose height
    # is the feed height padded up to the next /32 (1920×1080→1088,
    # 1280×720→736, 960×540→544, 640×360→384, 480×270→288, 320×180→192):
    # the canvas letterbox adds the pad rows at scale 1.0 (no resampling)
    # and postprocess crops them back, so the depth map / point cloud come
    # out exactly at the feed resolution. The 320×192 default (feed
    # 320×180) is first — it carries the family's single `preset:fast`
    # tag and is what "first variant" fallbacks pick.
    _EXPORT_SIZES: tuple[tuple[int, int], ...] = (
        (320, 192),
        (1920, 1088), (1280, 736), (960, 544), (640, 384), (480, 288),
    )

    @classmethod
    def variants(cls) -> tuple[InferenceVariant, ...]:
        # All resolutions share the same weights — only the baked input
        # dims differ. Tier numbers mirror LightStereo's fast variant
        # (same architecture class / footprint).
        out = []
        for w, h in cls._EXPORT_SIZES:
            default = (w, h) == (320, 192)
            tags = ["live", "low-latency" if w <= 640 else "high-res"]
            if default:
                tags.append("preset:fast")
            tags += ["weight:light", "year:2025"]
            out.append(InferenceVariant(
                slug=f"depz-stereo-{w}x{h}",
                label=f"{w}×{h}",
                files_to_download=(
                    ModelFile(
                        # The _sim ZIPs name the inner .onnx after the
                        # slug itself (verified pattern from
                        # LiteAnyStereo's live-CDN exports).
                        slug=f"depz_stereo_{w}x{h}_opset20_sim",
                        filename="model.onnx",
                        archive_member=(
                            f"depz_stereo_{w}x{h}_opset20_sim.onnx"),
                    ),
                ),
                configurations={
                    "size": "s", "input_size": (h, w),
                    "selector_meta": {
                        "task": "stereo",
                        "speed_tier": 5,
                        "accuracy_tier": 3,
                        "min_engine_class": "cpu",
                        "recommended_engine_class": "igpu",
                        "size_mb": 14.0,
                        "params_m": 3.44,
                        "published_fps": {"nvidia": 58},
                        "accuracy_metric": "SceneFlow EPE 0.73 px",
                        "summary": "Fast neural stereo, runs even on your CPU.",
                        "display_family": "DEPZ Stereo",
                        "tags": tags,
                        "neural": True,
                        "wizard_visible": True,
                        # DEPZ's own model — open, no external license
                        # terms to agree to.
                        "license": "DEPZ",
                        "license_url": "",
                        "min_ram_mb": 50,
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
