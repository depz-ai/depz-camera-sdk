"""Depth-domain filters: ZClip + Edge + TemporalEma.

Run after disparity → depth projection. ZClip is purely stateless;
EdgeFilter caches per-shape scratch buffers (cleared by
:meth:`reset`); TemporalEmaFilter holds the previous depth frame
(also cleared by :meth:`reset`).
"""

import numpy as np

from .base import BoolControl, DepthFilter, FloatControl


class ZClipFilter(DepthFilter):
    """Zero out depth pixels outside ``[min, max]`` meters.

    Destructive — downstream filters and the point cloud see the
    clipped depth. For a non-destructive view-only equivalent, the
    web viewer's render config exposes ``min_depth`` / ``max_depth``
    that drive colormap normalization + cloud near/far culling at
    render time without touching the underlying data.
    """

    SLUG = "zclip"
    LABEL = "Z-clip (drop out-of-range depth)"

    # ``min``/``max`` additionally maintain the ``max > min`` invariant
    # in :meth:`control_set`.
    enabled = BoolControl(default=False, label="Enable")
    min = FloatControl("number", default=0.3, min=0.05, max=20.0, step=0.05,
                    label="Min", unit="m")
    max = FloatControl("number", default=5.0, min=0.1, max=20.0, step=0.1,
                    label="Max", unit="m")

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return frame
        return np.where(
            (frame >= self.min) & (frame <= self.max),
            frame, np.float32(0.0),
        ).astype(np.float32, copy=False)

    def control_set(self, name: str, value):
        # max must stay strictly above min. The corrective write goes
        # straight to the backing attr — it's a clamp on our own
        # invariant, not a user value that needs coercing.
        v = super().control_set(name, value)
        if name in ("min", "max") and self.max <= self.min:
            self._store_control_raw("max", self.min + 0.01)
            if name == "max":
                return self.max
        return v


# ── Edge / discontinuity filter — has scratch buffers ─────────────────

class EdgeFilter(DepthFilter):
    """Zero out pixels at depth discontinuities (edge skirts / flyers).

    Keeps per-shape scratch buffers between calls so the per-frame
    work is allocation-free in steady state. Buffers re-allocate on
    canvas resize; :meth:`reset` clears them.
    """

    SLUG = "edge"
    LABEL = "Edge / discontinuity (drop boundary skirts)"

    enabled = BoolControl(default=False, label="Enable")
    rel_thresh = FloatControl("number", default=0.05, min=0.001, max=2.0,
                           step=0.005, label="|Δz| / z threshold")

    def __init__(self) -> None:
        super().__init__()
        # Keyed by (h, w) — supports canvas resize without trashing
        # buffers for the previous shape until next reset().
        self._bufs: dict[tuple[int, int], dict] = {}

    def reset(self) -> None:
        self._bufs.clear()

    def _get_bufs(self, shape: tuple[int, int]) -> dict:
        bufs = self._bufs.get(shape)
        if bufs is not None:
            return bufs
        bufs = {
            "valid": np.empty(shape, dtype=bool),
            "grad":  np.empty(shape, dtype=np.float32),
            "diff":  np.empty(shape, dtype=np.float32),
            "vmask": np.empty(shape, dtype=bool),
            "thr":   np.empty(shape, dtype=np.float32),
        }
        self._bufs[shape] = bufs
        return bufs

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return frame
        depth = frame
        h, w = depth.shape
        bufs = self._get_bufs((h, w))
        valid = bufs["valid"]
        grad = bufs["grad"]
        np.greater(depth, 0, out=valid)
        grad.fill(0)
        # Horizontal: |d[:,1:] - d[:,:-1]|, masked by both-sides-valid.
        if w >= 2:
            d_h = bufs["diff"][:, 1:]
            v_h = bufs["vmask"][:, 1:]
            np.subtract(depth[:, 1:], depth[:, :-1], out=d_h)
            np.abs(d_h, out=d_h)
            np.logical_and(valid[:, 1:], valid[:, :-1], out=v_h)
            np.multiply(d_h, v_h, out=d_h)
            np.maximum(grad[:, :-1], d_h, out=grad[:, :-1])
            np.maximum(grad[:, 1:],  d_h, out=grad[:, 1:])
        # Vertical
        if h >= 2:
            d_v = bufs["diff"][1:, :]
            v_v = bufs["vmask"][1:, :]
            np.subtract(depth[1:, :], depth[:-1, :], out=d_v)
            np.abs(d_v, out=d_v)
            np.logical_and(valid[1:, :], valid[:-1, :], out=v_v)
            np.multiply(d_v, v_v, out=d_v)
            np.maximum(grad[:-1, :], d_v, out=grad[:-1, :])
            np.maximum(grad[1:, :],  d_v, out=grad[1:, :])
        thr = bufs["thr"]
        np.maximum(depth, 1e-6, out=thr)
        np.multiply(thr, np.float32(self.rel_thresh), out=thr)
        drop = bufs["vmask"]
        np.greater(grad, thr, out=drop)
        np.logical_and(drop, valid, out=drop)
        out = depth.astype(np.float32, copy=True)
        out[drop] = np.float32(0.0)
        return out


# ── Temporal EMA — holds prev frame ───────────────────────────────────

class TemporalEmaFilter(DepthFilter):
    """Per-pixel EMA blending of depth with the previous frame.

    `alpha` weights the current frame: ``out = α·cur + (1-α)·prev``.
    α=1 is "off" (current passes through). Both pixels must be valid
    (>0) to blend; otherwise the current pixel passes through.
    """

    SLUG = "temporal_ema"
    LABEL = "Temporal EMA (smooth between frames)"

    # α=0.3 by default — heavy enough that toggling on/off is
    # immediately visible on a noisy depth map (α=0.5 was too subtle
    # and read as "the switch does nothing").
    enabled = BoolControl(default=False, label="Enable")
    alpha = FloatControl("number", default=0.3, min=0.05, max=1.0, step=0.05,
                      label="α (lower = stronger smoothing, 1=off)")

    def __init__(self) -> None:
        super().__init__()
        self._prev: np.ndarray | None = None

    def reset(self) -> None:
        self._prev = None

    def control_set(self, name: str, value):
        # When the user disables the filter, drop the cached previous
        # frame so re-enabling starts fresh — otherwise the next
        # frame would blend against an arbitrarily stale `_prev`,
        # producing a one-frame ghosting glitch.
        v = super().control_set(name, value)
        if name == "enabled" and not self.enabled:
            self._prev = None
        return v

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return frame.astype(np.float32, copy=False)
        depth = frame
        prev = self._prev
        if prev is None or prev.shape != depth.shape:
            self._prev = depth.copy()
            return depth.astype(np.float32, copy=False)
        alpha = np.float32(self.alpha)
        out = depth.copy()
        cur_v = depth > 0
        prev_v = prev > 0
        both = cur_v & prev_v
        out[both] = alpha * depth[both] + (1.0 - alpha) * prev[both]
        prev[cur_v] = out[cur_v]
        return out.astype(np.float32, copy=False)
