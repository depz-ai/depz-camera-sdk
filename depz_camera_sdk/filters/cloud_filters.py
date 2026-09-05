"""Point-cloud-domain filters.

Run after :meth:`Calibration.depth_to_point_cloud` produces the
organized cloud (h × w × 3 float32 in the calibration's metric
frame). Filters preserve the organized shape — out-of-region points
are zeroed rather than removed, so downstream consumers (renderer,
recorder) see the same array shape regardless of filter state. The
renderer already culls zero points.
"""

import numpy as np

from .base import BoolControl, FloatControl, IntControl, PointCloudFilter


class BoundingBoxFilter(PointCloudFilter):
    """Drop points outside an axis-aligned 3D box (in the calibration's
    metric frame, typically meters in the camera-rig frame).

    Strict superset of :class:`ZClipFilter` (which only handles the Z
    axis on depth). Useful when the consumer cares about a specific
    region of space — robotics workspace, AR table, etc.

    Out-of-box points are zeroed (XYZ all set to 0); the organized
    h×w×3 shape stays intact so the wire protocol is unchanged.
    """

    SLUG = "bbox"
    LABEL = "Bounding box (3D ROI)"

    # ``*_min``/``*_max`` additionally maintain the ``*_max > *_min``
    # invariant in :meth:`control_set`. Defaults span ±5 m horizontally /
    # vertically and 0.1–10 m in depth — a generous workspace box.
    enabled = BoolControl(default=False, label="Enable")
    x_min = FloatControl("number", default=-5.0, min=-20.0, max=20.0, step=0.05,
                      label="X min", unit="m")
    x_max = FloatControl("number", default=5.0, min=-20.0, max=20.0, step=0.05,
                      label="X max", unit="m")
    y_min = FloatControl("number", default=-5.0, min=-20.0, max=20.0, step=0.05,
                      label="Y min", unit="m")
    y_max = FloatControl("number", default=5.0, min=-20.0, max=20.0, step=0.05,
                      label="Y max", unit="m")
    z_min = FloatControl("number", default=0.1, min=0.05, max=20.0, step=0.05,
                      label="Z min", unit="m")
    z_max = FloatControl("number", default=10.0, min=0.1, max=20.0, step=0.1,
                      label="Z max", unit="m")

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return frame
        cloud = frame
        # `cloud` is h×w×3 float32. Build a per-pixel "in-box" mask
        # without materializing the full broadcast.
        x = cloud[..., 0]
        y = cloud[..., 1]
        z = cloud[..., 2]
        in_box = ((x >= self.x_min) & (x <= self.x_max)
                  & (y >= self.y_min) & (y <= self.y_max)
                  & (z >= self.z_min) & (z <= self.z_max))
        out = cloud.copy()
        # Broadcast the 2D mask across the trailing channel axis.
        out[~in_box] = np.float32(0.0)
        return out

    def control_set(self, name: str, value):
        # Maintain max > min on each axis (floor max to min+ε if the
        # user dragged max below min). Corrective writes bypass coerce
        # via _store_control_raw — they enforce our own invariant.
        v = super().control_set(name, value)
        eps = 0.01
        for axis in ("x", "y", "z"):
            mn = getattr(self, f"{axis}_min")
            mx = getattr(self, f"{axis}_max")
            if mx <= mn:
                if name == f"{axis}_min":
                    self._store_control_raw(f"{axis}_max", mn + eps)
                else:
                    # raising min above max while editing min: clamp the
                    # value we just set.
                    self._store_control_raw(f"{axis}_min", mx - eps)
                    if name == f"{axis}_min":
                        return mx - eps
        return v


# ── Statistical outlier removal ───────────────────────────────────────

# 8 grid neighbours — exhaustive 3×3 - center.
_GRID_OFFSETS = (
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
)


class StatisticalOutlierFilter(PointCloudFilter):
    """Drop points whose local 3D neighborhood is much sparser than
    the cloud's average — the classic "flying pixel" remover.

    For the organized cloud, "k nearest neighbors" reduces to the 8
    grid neighbors at (±1, ±1). For each non-zero pixel, we compute
    the mean Euclidean distance to its valid (non-zero) grid
    neighbors, then drop pixels whose mean distance exceeds the
    global ``μ + std_ratio · σ`` of those means.

    Catches points that survived :class:`EdgeFilter` (which only
    looks at depth gradients in 2D) but are isolated in 3D — typical
    flyers at object boundaries after stereo inference noise.

    Controls:
      - `std_ratio` — pixels with mean distance > μ + std_ratio·σ are
        outliers. Default 1.0 (aggressive). Higher = more permissive.
      - `min_neighbors` — pixels with fewer valid grid-neighbors are
        treated as outliers regardless of distance (corners /
        isolated patches).
    """

    SLUG = "stat_outlier"
    LABEL = "Statistical outlier (drop 3D flyers)"

    enabled = BoolControl(default=False, label="Enable")
    std_ratio = FloatControl("number", default=1.0, min=0.1, max=5.0, step=0.05,
                          label="σ ratio")
    min_neighbors = IntControl("number", default=3, min=1, max=8, step=1,
                            label="Min valid neighbours")

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return frame
        cloud = frame
        h, w = cloud.shape[:2]
        # A pixel is "valid" iff any XYZ component is non-zero. ZClip /
        # BBox / EdgeFilter all zero out invalid points, and
        # depth_to_point_cloud emits zeros for zero-depth pixels.
        valid = np.any(cloud != 0, axis=-1)

        sum_dist = np.zeros((h, w), dtype=np.float32)
        cnt = np.zeros((h, w), dtype=np.int32)

        for dy, dx in _GRID_OFFSETS:
            # Slices for self / shifted-neighbour, accounting for
            # whichever direction the offset goes.
            sy = slice(max(0, -dy), h - max(0,  dy))
            sx = slice(max(0, -dx), w - max(0,  dx))
            ny = slice(max(0,  dy), h - max(0, -dy))
            nx = slice(max(0,  dx), w - max(0, -dx))
            diff = cloud[sy, sx] - cloud[ny, nx]
            d = np.sqrt(np.einsum("...i,...i->...", diff, diff)
                        ).astype(np.float32, copy=False)
            both = valid[sy, sx] & valid[ny, nx]
            sum_dist[sy, sx] += np.where(both, d, np.float32(0.0))
            cnt[sy, sx] += both

        # Mean distance per pixel, with cnt=0 → 0 (and we'll mask
        # those out below as "too few neighbours").
        mean_d = sum_dist / np.maximum(cnt, 1)

        # Two drop conditions, both gated by "the pixel was valid":
        #   1. fewer valid grid-neighbours than `min_neighbors` —
        #      definitionally suspicious; drop regardless of stats.
        #   2. mean distance to neighbours > μ + n·σ over the
        #      well-supported sample — needs a non-empty population.
        has_enough = cnt >= self.min_neighbors
        out = cloud.copy()
        drop = valid & ~has_enough

        sample = mean_d[valid & has_enough]
        if sample.size > 0:
            thresh = float(sample.mean()) + self.std_ratio * float(sample.std())
            drop = drop | (valid & has_enough & (mean_d > thresh))
        out[drop] = np.float32(0.0)
        return out
