"""Depth/disparity/cloud post-processing filter chain.

One :class:`Filter` ABC; each concrete filter is self-describing
(declares its own parameters via :class:`Control`) so the UI
can build itself from server introspection without hardcoding any
filter controls.

Default filter chain (order matters — pipeline applies all
``OPERATES_ON="disparity"`` filters before depth conversion, then all
``OPERATES_ON="depth"`` filters after, then all
``OPERATES_ON="point_cloud"`` filters after the depth → cloud
projection):

  1. :class:`SpeckleFilter`       — disparity, stateless
  2. ``disparity → depth``        (calibration step, lives outside this module)
  3. :class:`ZClipFilter`         — depth, stateless
  4. :class:`EdgeFilter`          — depth, scratch buffers
  5. :class:`TemporalEmaFilter`   — depth, holds prev-frame state
  6. ``depth → point_cloud``      (calibration step, lives outside this module)
  7. :class:`BoundingBoxFilter`        — point_cloud, stateless
  8. :class:`StatisticalOutlierFilter` — point_cloud, stateless
"""

from .base import (
    DepthFilter, DisparityFilter, Filter, Control, PointCloudFilter,
)
from .cloud_filters import BoundingBoxFilter, StatisticalOutlierFilter
from .depth_filters import EdgeFilter, TemporalEmaFilter, ZClipFilter
from .disparity_filters import SpeckleFilter


def default_filter_chain() -> list[Filter]:
    """The pipeline's stock filter chain. Construction order matches
    application order — ``OPERATES_ON`` is the only thing the worker
    looks at when partitioning into the disparity / depth / point-cloud
    domains."""
    return [
        SpeckleFilter(),
        ZClipFilter(), EdgeFilter(), TemporalEmaFilter(),
        BoundingBoxFilter(), StatisticalOutlierFilter(),
    ]


__all__ = [
    "Filter", "Control",
    "DisparityFilter", "DepthFilter", "PointCloudFilter",
    "SpeckleFilter",
    "ZClipFilter", "EdgeFilter", "TemporalEmaFilter",
    "BoundingBoxFilter", "StatisticalOutlierFilter",
    "default_filter_chain",
]
