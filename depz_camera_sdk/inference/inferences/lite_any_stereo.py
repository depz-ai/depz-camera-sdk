"""LiteAnyStereoInference — fast zero-shot stereo matching (LAS V2).

Network: 2 inputs (left, right BGR canvas), 1 output (canvas-sized
disparity in pixels). CVPR 2026, Imperial College London.
MIT license — commercial use permitted.

Preprocess:
- BGR uint8 → RGB float32 in [0, 255] — no external normalisation.
  The ONNX export applies ImageNet mean/std internally in its forward pass.
- One ``cv2.dnn.blobFromImageWithParams`` call per eye.

Postprocess:
- Squeeze to (H, W).
- Crop to the valid canvas region (strips letterbox padding).
- Cast to float32 contiguous.
"""

import cv2
import numpy as np

from ...calibration import CanvasLayout
from ..base import InferenceVariant, ModelFile, NeuralStereoInferenceBase
from ..registry import register_inference


@register_inference("lite-any-stereo")
class LiteAnyStereoInference(NeuralStereoInferenceBase):
    """LiteAnyStereo V2 S — fast disparity stereo. [-1, 1] normalisation,
    squeeze-and-crop postprocess. ONNX export has concrete H×W dims so
    no INPUT_SIZE override is needed."""

    LABEL = "Lite Any Stereo"

    # (W, H) of every concrete ONNX export (the `_sim`-simplified opset20
    # builds). Each camera feed step has a SAME-WIDTH export whose height
    # is the feed height padded up to the next /32 (1920×1080→1088,
    # 1280×720→736, 960×540→544, 640×360→384, 480×270→288, 320×180→192):
    # the canvas letterbox adds the pad rows at scale 1.0 (no resampling)
    # and postprocess crops them back, so the depth map / point cloud come
    # out exactly at the feed resolution. The 640×384 default is first —
    # it carries the family's single `preset:balanced` tag and is what
    # "first variant" fallbacks pick.
    _EXPORT_SIZES: tuple[tuple[int, int], ...] = (
        (640, 384),
        (1920, 1088), (1280, 736), (960, 544), (480, 288), (320, 192),
    )

    @classmethod
    def variants(cls) -> tuple[InferenceVariant, ...]:
        # Tier source: CVPR 2026 paper (arXiv 2606.24457). Ranks 1st across
        # ETH3D / KITTI 2012+2015 / Middlebury / DrivingStereo zero-shot.
        # Speed numbers not officially published for the S variant; the
        # heuristic speed_tier=5 reflects the lightweight architecture
        # (comparable footprint to LightStereo S). All resolutions share
        # the same weights — only the baked input dims differ.
        out = []
        for w, h in cls._EXPORT_SIZES:
            default = (w, h) == (640, 384)
            tags = ["live", "low-latency" if w <= 640 else "high-res"]
            if default:
                tags.append("preset:balanced")
            tags += ["weight:light", "year:2025"]
            out.append(InferenceVariant(
                slug=f"v2-s-{w}x{h}",
                label=f"V2 S / {w}×{h}",
                files_to_download=(
                    ModelFile(
                        # The _sim ZIPs name the inner .onnx after the
                        # slug itself (verified against the live CDN) —
                        # unlike the retired LAS2_S_* convention.
                        slug=f"lite_any_stereo_v2_S_{w}x{h}_opset20_sim",
                        filename="model.onnx",
                        archive_member=(
                            f"lite_any_stereo_v2_S_{w}x{h}_opset20_sim.onnx"),
                    ),
                ),
                configurations={
                    "size": "s", "input_size": (h, w),
                    "selector_meta": {
                        "task": "stereo",
                        "speed_tier": 5,
                        "accuracy_tier": 4,
                        "min_engine_class": "cpu",
                        "recommended_engine_class": "igpu",
                        "size_mb": 14.0,
                        "published_fps": {},
                        "accuracy_metric": (
                            "Ranks 1st across ETH3D / KITTI / Middlebury / "
                            "DrivingStereo (zero-shot, CVPR 2026)"
                        ),
                        "summary": "Modern fast stereo.",
                        "display_family": "Lite Any Stereo v2.0 · S",
                        "tags": tags,
                        "license": "MIT",
                        "license_url": (
                            "https://github.com/TomTomTommi/LiteAnyStereo"
                            "/blob/main/LICENSE"
                        ),
                        "neural": True,
                        "wizard_visible": True,
                        "min_ram_mb": 100,
                    },
                },
            ))
        return tuple(out)

    def _init_extra_state(self) -> None:
        # LAS2 ONNX export normalises internally (ImageNet stats baked in
        # as a normalize_image() call in the forward pass). Feed raw BGR→RGB
        # uint8 as float32 in [0, 255] — no external mean/std needed.
        self._blob_params = cv2.dnn.Image2BlobParams(
            scalefactor=(1.0, 1.0, 1.0, 0.0),
            size=(0, 0),
            mean=(0.0, 0.0, 0.0, 0.0),
            swapRB=True,
            ddepth=cv2.CV_32F,
        )

    def preprocess(
        self,
        canvas_l: np.ndarray,
        canvas_r: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        L = cv2.dnn.blobFromImageWithParams(canvas_l, self._blob_params)
        R = cv2.dnn.blobFromImageWithParams(canvas_r, self._blob_params)
        return L, R

    def postprocess(
        self,
        raw_output: np.ndarray,
        canvas_layout: CanvasLayout,
    ) -> np.ndarray:
        arr = np.asarray(raw_output)
        disp = np.squeeze(arr)
        if disp.ndim != 2:
            raise RuntimeError(
                f"model output shape {tuple(arr.shape)} does not yield "
                f"a 2D map after squeeze (got {tuple(disp.shape)})"
            )
        if disp.dtype != np.float32:
            disp = disp.astype(np.float32, copy=False)
        y, x = canvas_layout.pad_y, canvas_layout.pad_x
        h, w = canvas_layout.valid_h, canvas_layout.valid_w
        return np.ascontiguousarray(disp[y: y + h, x: x + w])
