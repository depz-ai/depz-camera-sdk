"""Filter ABC + Control descriptor.

A :class:`Filter` is one self-describing post-processing stage that
runs on either the disparity map or the depth map. Each filter owns
its own state + parameters; the pipeline composes them into a list
and walks them once per frame.

Concrete filters subclass :class:`Filter`, set the three class
attributes (``KIND`` / ``LABEL`` / ``OPERATES_ON``), declare their
controls in a ``_CONTROL_SPECS`` class tuple, implement ``apply()``, and
override ``control_set()`` only when a parameter needs custom coercion
(e.g. snapping to a discrete set).

There's no separate "stateful filter" ABC — :meth:`Filter.reset` is
a no-op by default; stateful filters override it.

The descriptor (:class:`Control`) and the get/set/coerce
machinery are shared with :class:`~depz_camera_sdk.inference.base.DepthInferenceBase`
through :class:`~depz_camera_sdk.._controls.ControlOwner`. ``Control``
is a re-export of the common :class:`~depz_camera_sdk.._controls.Control`."""

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from .._controls import BoolControl, FloatControl, IntControl, Control, ControlOwner, StrControl

__all__ = [
    "Control", "BoolControl", "IntControl", "FloatControl", "StrControl",
    "Filter",
    "DisparityFilter",
    "DepthFilter",
    "PointCloudFilter",
]


class Filter(ABC, ControlOwner):
    """One depth/disparity post-processing stage.

    Lifecycle (per pipeline):
      - construction: filter holds default param values, ``enabled=False``.
      - per-frame: pipeline calls ``apply(frame)`` (no-op when disabled).
      - on seek / canvas resize / disable→enable flip: pipeline calls
        ``reset()``.
      - param change: caller goes through pipeline command queue, which
        ultimately calls ``set(name, value)`` on the filter from the
        worker thread.

    Subclass contract:
      - set ``SLUG`` (canonical id, e.g. ``"median"``) — the stable
        wire identifier for RPC routing + UI introspection. Same role
        as engine slugs (``"openvino-CPU"``).
      - inherit from one of :class:`DisparityFilter` /
        :class:`DepthFilter` / :class:`PointCloudFilter` to declare
        ``OPERATES_ON``; do not set ``OPERATES_ON`` directly.
      - optional ``LABEL`` (defaults to ``SLUG`` if empty).
      - declare ``_CONTROL_SPECS`` (a tuple of :class:`Control`); its first
        entry MUST be the ``"enabled"`` boolean. Read-only properties
        and default seeding are generated from it.
      - implement ``apply()``.

    ``control_get``/``control_set``/``controls_to_dict`` and the default
    coerce-clamp-snap logic come from
    :class:`~depz_camera_sdk.._controls.ControlOwner`; subclasses override
    ``control_set`` only for custom snapping (discrete sets,
    cross-parameter constraints).
    """

    SLUG: ClassVar[str] = ""
    LABEL: ClassVar[str] = ""
    OPERATES_ON: ClassVar[str] = ""

    def __init__(self) -> None:
        # Seed every backing attribute from its descriptor default.
        # Subclasses with extra (non-control) state override __init__,
        # call super().__init__(), then initialize that state.
        self._seed_control_defaults()

    # ── Public methods ────────────────────────────────────────────────

    @abstractmethod
    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Run the filter on `frame`. Return a new ndarray (callers
        may retain it). Must early-return when ``self.enabled`` is
        False."""

    def reset(self) -> None:
        """Clear frame-to-frame state. Default no-op (most filters
        have nothing to reset). Stateful filters override."""

    # Control descriptors are introspected via the inherited
    # :meth:`ControlOwner.controls`. First entry is the ``"enabled"`` boolean.


# ── Intermediate ABCs — one per processing domain ─────────────────────
#
# Concrete filters subclass DisparityFilter / DepthFilter /
# PointCloudFilter to declare which stage of the pipeline they slot
# into. The intermediate ABC pre-fills :attr:`Filter.OPERATES_ON`, so
# concrete classes don't repeat it. The pipeline / postprocessor can
# also do `isinstance(f, DepthFilter)` for type-checked routing if
# they need to.

class DisparityFilter(Filter):
    """Runs on the raw disparity map (h1 × w1 float32 pixels) before
    the calibration projects disparity → depth."""

    OPERATES_ON = "disparity"


class DepthFilter(Filter):
    """Runs on the depth map (h1 × w1 float32 meters) after disparity
    → depth projection but before depth → cloud projection."""

    OPERATES_ON = "depth"


class PointCloudFilter(Filter):
    """Runs on the organized point cloud (h1 × w1 × 3 float32 meters,
    in the calibration's metric frame) after depth → cloud projection.

    Filters here preserve the organized h×w×3 shape so the wire
    protocol stays stable; out-of-region points are zeroed rather
    than removed. Downstream consumers (renderer, recorder) skip
    zero points."""

    OPERATES_ON = "point_cloud"
