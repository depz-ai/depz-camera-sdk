"""DepzStereoInference — DEPZ's own fast stereo disparity network.

Network: 2 inputs (left, right canvas), 1 output (canvas-sized disparity
in pixels). Same input recipe (ImageNet-normalised RGB) and output shape
as LightStereo — both are the :class:`NeuralStereoInferenceBase`
defaults; only the weights and the resolution lineup differ.
"""

from ..base import (
    IMAGENET_BLOB_MEAN, IMAGENET_BLOB_SCALE, InferenceVariant,
    NeuralStereoInferenceBase, onnx_variant,
)
from ..registry import register_inference
from ..resolution_meta import resolution_meta


# sha256 / size of each extracted .onnx (the file inside the ZIP — that
# is what ModelFile verifies), keyed by (W, H).
_FILE_FACTS: dict[tuple[int, int], tuple[str, int]] = {
    (320, 192): ("61c3bc796227cbd858e47a4b2ac5ee16f4e3c583cd778f25d7a2b6977f9fec68", 22515405),
    (480, 288): ("3c97b9e655c34c93d9e58889898fbe2a3f77117c03800518b955007373185231", 22515431),
    (640, 384): ("e3ff9bf4c55695300907cafb38b1aeb4c996419ee2553afb2d4313e7fd6b2a77", 22515738),
    (960, 544): ("86c30670651375f1821c18990c28b5386f530745458a1e3678686ee8497c13ab", 22516046),
    (1280, 736): ("00667228e0d67b044f1c63fb782ed70c7d0c79bd1490065f44ad06e3b4dd5169", 22516125),
    (1920, 1088): ("2862b83b202a6b9761648a450c168605663adeed71e6d041a983160500722c36", 22516204),
}


@register_inference("depz-stereo")
class DepzStereoInference(NeuralStereoInferenceBase):
    """DEPZ Stereo — fast disparity stereo, DEPZ's own model.
    ImageNet-normalised input; the ONNX exports pin (H, W) in the
    graph."""

    LABEL = "DEPZ Stereo"
    BLOB_MEAN = IMAGENET_BLOB_MEAN
    BLOB_SCALE = IMAGENET_BLOB_SCALE

    # (W, H) of every concrete ONNX export (the `_sim`-simplified opset20
    # builds). Each camera feed step has a SAME-WIDTH export whose height
    # is the feed height padded up to the next /32 (1920×1080→1088,
    # 1280×720→736, 960×540→544, 640×360→384, 480×270→288, 320×180→192):
    # the canvas letterbox adds the pad rows at scale 1.0 (no resampling)
    # and postprocess crops them back, so the depth map / point cloud come
    # out exactly at the feed resolution. The 320×192 default (feed
    # 320×180) is first — it is what "first variant" fallbacks pick.
    # No preset tag: the SimpleViewer's Fast slot belongs to Lite Any
    # Stereo V2 S (lite_any_stereo.py); this family stays selectable
    # from the Advanced model list.
    _EXPORT_SIZES: tuple[tuple[int, int], ...] = (
        (320, 192),
        (1920, 1088), (1280, 736), (960, 544), (640, 384), (480, 288),
    )

    @classmethod
    def _variants(cls) -> tuple[InferenceVariant, ...]:
        # All resolutions share the same weights — only the baked input
        # dims differ. Same building blocks as LightStereo-S, widened
        # (5.58M parameters, 22.5 MB) with a third refinement stage.
        out = []
        for w, h in cls._EXPORT_SIZES:
            tags = ["live", "low-latency" if w <= 640 else "high-res",
                    "weight:light", "year:2025"]
            stem = f"depz_stereo_{w}x{h}_opset20_sim"
            sha256, size_bytes = _FILE_FACTS[w, h]
            out.append(onnx_variant(
                slug=f"depz-stereo-{w}x{h}",
                label=f"{w}×{h}",
                input_size=(h, w),
                model_slug=stem,
                # The _sim ZIPs name the inner .onnx after the slug.
                archive_member=f"{stem}.onnx",
                sha256=sha256,
                size_bytes=size_bytes,
                selector_meta={
                    "task": "stereo",
                    "speed_tier": 5,
                    "accuracy_tier": 3,
                    "min_engine_class": "cpu",
                    "recommended_engine_class": "igpu",
                    "size_mb": 22.5,
                    "params_m": 5.58,
                    "published_fps": {},
                    "accuracy_metric": "",
                    "summary": "Fast neural stereo, runs even on your CPU.",
                    "display_family": "DEPZ Stereo",
                    "tags": tags,
                    "neural": True,
                    # DEPZ's own model — open, no external license
                    # terms to agree to.
                    "license": "DEPZ",
                    "license_url": "",
                    "min_ram_mb": 50,
                    # Measured with tools/model_eval. No size is marked
                    # verified: on the benchmarks the model loses whole
                    # scenes at every size and degrades above 960.
                    # Disparity ceiling: the network is built for 192 px,
                    # but with the right image shifted to probe larger
                    # disparities its accuracy falls from ~150 px and is
                    # gone by ~170 (640 wide: 84 % of pixels within 5 %
                    # at 100-150 px, 58 % at 150-188; 320 wide: 74 % at
                    # 130-160, 13 % at 160-200) — so 160, not 192.
                    **resolution_meta(
                        w, max_disparity_px=160.0,
                        error_px={320: 0.39, 480: 0.38, 640: 0.39,
                                  960: 0.55, 1280: 0.72, 1920: 1.41},
                        verified_widths=frozenset()),
                },
            ))
        return tuple(out)
