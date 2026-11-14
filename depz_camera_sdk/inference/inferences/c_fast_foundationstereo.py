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

Input (confirmed against the NVlabs source): the export strips ImageNet
normalisation from the graph, so the caller applies it —
``(pixel/255 - mean) / std``, RGB order. Output: one (1, 1, H, W)
disparity map. Both are the :class:`NeuralStereoInferenceBase` defaults.
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
    (320, 192): ("7ecae8d1fd340442fc69e1359902d8bbfad69a83681cf7fe9a0216613fd867a0", 59813098),
    (480, 288): ("c2e223386f7d8a705bbe9b71d1071684c95fb6fee67147d8b6b1bbc9bcfc3fd3", 60947669),
    (640, 384): ("c810494f1a34604291dcc984a6e63b3d6f66e68a7ac98ffd7c57f5659d488843", 62536379),
    (960, 544): ("9b40e8a572bd519dcd0892697ff8460a1fa681c181c4bca3af95fafc420480e1", 66618976),
    (1280, 736): ("bbab585bcd11783b3128240c1c958f9039cba22993e35be76ad87d8ea7926566", 72814786),
    (1920, 1088): ("346b48ae4e12107f9e82306b04d4b9e61bcf3c889de587b42b7697597d4108a6", 89737338),
}


@register_inference("c-fast-foundationstereo")
class CFastFoundationStereoInference(NeuralStereoInferenceBase):
    """C-Fast-FoundationStereo — real-time zero-shot stereo disparity.
    ImageNet-normalised input (the export strips in-graph
    normalisation). The ONNX exports pin (H, W) in the graph and emit a
    single ``disparity`` output."""

    LABEL = "C-Fast-FoundationStereo"
    BLOB_MEAN = IMAGENET_BLOB_MEAN
    BLOB_SCALE = IMAGENET_BLOB_SCALE

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
    def _variants(cls) -> tuple[InferenceVariant, ...]:
        # All resolutions share the same weights + preprocess — only the
        # baked input dims differ. Every export runs 4 refinement
        # iterations and is built for 320 px of disparity.
        out = []
        for w, h in cls._EXPORT_SIZES:
            default = (w, h) == (640, 384)
            tags = ["high-res" if w >= 960 else "low-res",
                    "weight:heavy", "year:2025"]
            if default:
                tags = ["preset:accurate", "weight:heavy", "year:2025"]
            stem = f"c_ffs_{w}x{h}_opset17_sim"
            sha256, size_bytes = _FILE_FACTS[w, h]
            out.append(onnx_variant(
                slug=f"iters4-{w}x{h}",
                label=f"iters 4 / {w}×{h}",
                input_size=(h, w),
                model_slug=stem,
                # The _sim ZIPs name the inner .onnx after the slug.
                archive_member=f"{stem}.onnx",
                sha256=sha256,
                size_bytes=size_bytes,
                selector_meta={
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
                    # Measured with tools/model_eval. The export is
                    # built for 320 px of disparity. 1920×1088 needs
                    # more than 12 GB of GPU memory.
                    **resolution_meta(
                        w, max_disparity_px=320.0,
                        error_px={320: 0.11, 480: 0.12, 640: 0.12,
                                  960: 0.16, 1280: 0.17},
                        verified_widths={320, 480, 640, 960, 1280},
                        notes={1920: "Needs more than 12 GB of GPU "
                                     "memory; falls back to the CPU "
                                     "otherwise."}),
                    "license": "NVIDIA Open Model Agreement",
                    "license_url": "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/",
                },
            ))
        return tuple(out)
