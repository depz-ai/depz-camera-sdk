"""Per-resolution facts about a stereo model, measured on public
benchmarks, in the form the catalog publishes them.

Every exported network is a fixed-size graph of the same weights, but a
model does not behave the same at every size: the largest disparity it
can output is a fixed number of pixels, and so is its typical error. In
metres both depend on the camera, so the catalog stores the pixel
figures and the consumer converts them with the camera's own geometry
(see :func:`working_range_m`).

The numbers come from ``tools/model_eval`` — Middlebury 2014 / 2021 and
ETH3D, 66 scenes, framed the way the pipeline frames camera input.
"""
from __future__ import annotations

from typing import Mapping, Optional

# Depth is called usable up to the distance where the model's typical
# disparity error becomes this share of the disparity (= of the depth).
RANGE_ERROR_SHARE = 0.05


def resolution_meta(width: int, *, max_disparity_px: float,
                    error_px: Mapping[int, float],
                    verified_widths: frozenset[int] | set[int],
                    notes: Optional[Mapping[int, str]] = None) -> dict:
    """``selector_meta`` entries describing one export width.

    * ``max_disparity_px`` — the largest disparity the network outputs;
      anything nearer than the matching distance comes out wrong.
    * ``disparity_error_px`` — median absolute error on the benchmarks
      at this width.
    * ``resolution_status`` — ``"verified"`` when the size was measured
      and behaves as the model does at its native resolution; absent
      otherwise (the viewer shows it as experimental).
    * ``resolution_note`` — why a size is not verified, for a tooltip.
    """
    out: dict = {"max_disparity_px": float(max_disparity_px)}
    if width in error_px:
        out["disparity_error_px"] = float(error_px[width])
    if width in verified_widths:
        out["resolution_status"] = "verified"
    if notes and width in notes:
        out["resolution_note"] = notes[width]
    return out


def working_range_m(width: int, range_factor_m: float, *,
                    max_disparity_px: Optional[float],
                    disparity_error_px: Optional[float],
                    ) -> tuple[Optional[float], Optional[float]]:
    """``(nearest, farthest)`` usable distance in metres for a model fed
    ``width``-pixel-wide frames from a camera whose
    ``range_factor_m = focal_px · baseline_m / frame_width_px`` (the
    factor does not depend on the feed size — focal length scales with
    the frame).

    Nearest: the distance whose disparity equals the model's ceiling.
    Farthest: where the typical error reaches
    :data:`RANGE_ERROR_SHARE` of the depth, in benchmark conditions — a
    real camera adds sensor noise and compression, so read it as an
    upper bound. ``None`` where the input is unknown.
    """
    fb = range_factor_m * width                       # focal·baseline, px·m
    near = fb / max_disparity_px if max_disparity_px else None
    far = (RANGE_ERROR_SHARE * fb / disparity_error_px
           if disparity_error_px else None)
    return near, far
