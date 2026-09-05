"""Disparity-domain filters: Speckle.

Runs on the raw disparity map produced by the model, before
disparity → depth projection. Stateless — :meth:`Filter.reset` is
the inherited no-op.
"""

import cv2
import numpy as np

from .base import BoolControl, DisparityFilter, FloatControl, IntControl


# ── Speckle ───────────────────────────────────────────────────────────

# filterSpeckles requires int16; we scale float disparity by 16 (4
# fractional bits) for the round-trip — matches OpenCV StereoBM
# convention.
_SPECKLE_SCALE = 16


class SpeckleFilter(DisparityFilter):
    """OpenCV's filterSpeckles — drops tiny disconnected components.

    `max_size` is the largest island (in pixels) that gets zeroed.
    `max_diff` is the connectivity threshold in disparity units.
    """

    SLUG = "speckle"
    LABEL = "Speckle (disparity)"

    # Controls declared once as typed descriptors. Reading e.g.
    # ``self.max_size`` is typed (int) and read-only; ControlOwner seeds
    # the defaults and collects these into the catalog.
    enabled = BoolControl(default=False, label="Enable")
    max_size = IntControl("number", default=50, min=1, max=2000, step=1,
                       label="Max island size", unit="px")
    max_diff = FloatControl("number", default=1.0, min=0.0, max=10.0, step=0.1,
                         label="Max neighbor Δ", unit="disp")

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return frame
        scale = _SPECKLE_SCALE
        scaled = np.clip(frame * scale, -32767, 32767).astype(np.int16)
        cv2.filterSpeckles(
            scaled, 0, self.max_size, int(round(self.max_diff * scale)),
        )
        out = frame.copy()
        out[scaled == 0] = np.float32(0.0)
        return out
