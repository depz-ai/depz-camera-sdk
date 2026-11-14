"""LiteAnyStereoInference — fast zero-shot stereo matching (LAS V2).

Network: 2 inputs (left, right BGR canvas), 1 output (canvas-sized
disparity in pixels). CVPR 2026, Imperial College London.
MIT license — commercial use permitted.

Input: raw RGB in [0, 255] — the ONNX export applies ImageNet mean/std
inside its forward pass. Output: one disparity map. Both are the
:class:`NeuralStereoInferenceBase` defaults.
"""

from ..base import InferenceVariant, NeuralStereoInferenceBase, onnx_variant
from ..registry import register_inference
from ..resolution_meta import resolution_meta


# sha256 / size of each extracted .onnx (the file inside the ZIP — that
# is what ModelFile verifies), keyed by (size, W, H).
_FILE_FACTS: dict[tuple[str, int, int], tuple[str, int]] = {
    ("S", 320, 192): ("73938b3a5ba2361a6ebbbfc957170d1d8015a94c7acd104da02e96bd7eaad9a2", 24310481),
    ("S", 480, 288): ("150ed1da1adf3cf6f1d5b6600558efa3c854ade694f312e4e769d625a8973848", 24310492),
    ("S", 640, 384): ("075c63ebe7eb0508ed815a8e271a2fc93fdc7c6aca2d707adc8780a75fde3d26", 24310671),
    ("S", 960, 544): ("b734ecec5f541541e55cf5262b4843f0ecf87589bf4b896b4761e572898dd282", 24310851),
    ("S", 1280, 736): ("1541f9391df67266ba8b7ba290a6ede8f1da597948dfa8a39d4b70e21b622410", 24310938),
    ("S", 1920, 1088): ("ae64ddbbf81d506855adfbb6d62c43b181aba7f882ce56f18bc3c6d6039a6a0e", 24311025),
    ("M", 320, 192): ("04fe3c0275f85a4d93540127be16446d4935e42f833d8b47550e2ea418e12bdf", 41687204),
    ("M", 480, 288): ("c94244c829076231bffb74bd5406af5eb0f6841354b85adb68180685193c7043", 41687214),
    ("M", 640, 384): ("abdbd11593ac51c07f1892d4d93a44cd28c7ca8922629bc4e618478dc63abc96", 41687416),
    ("M", 960, 544): ("e30ed28a659bdd5b1dd07fa1324e8ce8482cec97d922c73ebbd358f0582f82a2", 41687619),
    ("M", 1280, 736): ("a0483a49ecfc3d506530106bd35f0d686531240a763838b6cf3b558a59dba351", 41687751),
    ("M", 1920, 1088): ("8e65f6313dc4cd9f4e8807cd2e521141354a094892f75a014b76c7743a411828", 41687883),
}


@register_inference("lite-any-stereo")
class LiteAnyStereoInference(NeuralStereoInferenceBase):
    """LiteAnyStereo V2 S — fast disparity stereo. Takes RGB 0..255 (the
    network normalises internally); the ONNX exports have concrete H×W
    dims.

    The M size is :class:`LiteAnyStereoMInference` — a separate family,
    because the viewer keys the active preset card, its resolution list
    and the chosen resolution by family."""

    LABEL = "Lite Any Stereo"

    # Model size letter (upstream LAS2 S / M) and what differs per size.
    _SIZE = "S"
    _DEFAULT_SIZE = (320, 192)        # first variant; carries the preset tag
    _PRESET_TAG = "preset:fast"
    _SPEED_TIER = 5
    _ACCURACY_TIER = 4
    _SIZE_MB = 24.3
    _MIN_RAM_MB = 100
    _SUMMARY = "Modern fast stereo."

    # Measured with tools/model_eval (see resolution_meta.py). The network
    # outputs at most 188 px of disparity at any size, so the two largest
    # exports cannot see near objects (closer than ~0.6 m / ~0.9 m on a
    # Varius) and are left unverified.
    _MAX_DISPARITY_PX = 188.0
    _ERROR_PX = {320: 0.22, 480: 0.22, 640: 0.23, 960: 0.29,
                 1280: 0.34, 1920: 0.43}
    _VERIFIED_WIDTHS = frozenset({320, 480, 640, 960})
    _NOTES = {
        1280: "Near objects are cut off at this size — check the minimum "
              "distance.",
        1920: "Near objects are cut off at this size and overall accuracy "
              "is lower than at 960 — use only for distant scenes.",
    }

    # (W, H) of every concrete ONNX export (the `_sim`-simplified opset20
    # builds). Each camera feed step has a SAME-WIDTH export whose height
    # is the feed height padded up to the next /32 (1920×1080→1088,
    # 1280×720→736, 960×540→544, 640×360→384, 480×270→288, 320×180→192):
    # the canvas letterbox adds the pad rows at scale 1.0 (no resampling)
    # and postprocess crops them back, so the depth map / point cloud come
    # out exactly at the feed resolution.
    _EXPORT_SIZES: tuple[tuple[int, int], ...] = (
        (1920, 1088), (1280, 736), (960, 544), (640, 384), (480, 288),
        (320, 192),
    )

    @classmethod
    def _variants(cls) -> tuple[InferenceVariant, ...]:
        # Tier source: arXiv 2606.24457. All resolutions of one size share
        # the same weights — only the baked input dims differ. The default
        # is declared first: it carries the family's single preset tag and
        # is what "first variant" fallbacks pick.
        size, lower = cls._SIZE, cls._SIZE.lower()
        ordered = [cls._DEFAULT_SIZE] + [
            s for s in cls._EXPORT_SIZES if s != cls._DEFAULT_SIZE]
        out = []
        for w, h in ordered:
            tags = ["live", "low-latency" if w <= 640 else "high-res"]
            if (w, h) == cls._DEFAULT_SIZE:
                tags.append(cls._PRESET_TAG)
            tags += ["weight:light", "year:2026"]
            stem = f"lite_any_stereo_v2_{size}_{w}x{h}_opset20_sim"
            sha256, size_bytes = _FILE_FACTS[size, w, h]
            out.append(onnx_variant(
                slug=f"v2-{lower}-{w}x{h}",
                label=f"V2 {size} / {w}×{h}",
                input_size=(h, w),
                model_slug=stem,
                # The _sim ZIPs name the inner .onnx after the slug.
                archive_member=f"{stem}.onnx",
                sha256=sha256,
                size_bytes=size_bytes,
                selector_meta={
                    "task": "stereo",
                    "speed_tier": cls._SPEED_TIER,
                    "accuracy_tier": cls._ACCURACY_TIER,
                    "min_engine_class": "cpu",
                    "recommended_engine_class": "igpu",
                    "size_mb": cls._SIZE_MB,
                    "published_fps": {},
                    "accuracy_metric": (
                        "Ranks 1st across ETH3D / KITTI / Middlebury / "
                        "DrivingStereo (zero-shot, efficient models)"
                    ),
                    "summary": cls._SUMMARY,
                    "display_family": f"Lite Any Stereo v2.0 · {size}",
                    "tags": tags,
                    "license": "MIT",
                    "license_url": (
                        "https://github.com/TomTomTommi/LiteAnyStereo"
                        "/blob/main/LICENSE"
                    ),
                    "neural": True,
                    "min_ram_mb": cls._MIN_RAM_MB,
                    **resolution_meta(
                        w, max_disparity_px=cls._MAX_DISPARITY_PX,
                        error_px=cls._ERROR_PX,
                        verified_widths=cls._VERIFIED_WIDTHS,
                        notes=cls._NOTES),
                },
            ))
        return tuple(out)


@register_inference("lite-any-stereo-m")
class LiteAnyStereoMInference(LiteAnyStereoInference):
    """LiteAnyStereo V2 M — the same network with a deeper cost
    aggregation: about a third slower than S and clearly more accurate
    (Middlebury zero-shot Bad2.0 5.47 vs 8.87 in the paper). Same input
    and output contract as S."""

    LABEL = "Lite Any Stereo M"

    _SIZE = "M"
    _DEFAULT_SIZE = (640, 384)
    _PRESET_TAG = "preset:balanced"
    _SPEED_TIER = 4
    _ACCURACY_TIER = 4
    _SIZE_MB = 41.7
    _MIN_RAM_MB = 170
    _SUMMARY = "Modern stereo, more accurate than S."
    _ERROR_PX = {320: 0.22, 480: 0.20, 640: 0.21, 960: 0.28,
                 1280: 0.34, 1920: 0.42}
