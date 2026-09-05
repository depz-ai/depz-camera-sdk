"""Shared control descriptor + ``ControlOwner`` mixin.

Used by every class that exposes a list of self-describing runtime
parameters to the UI: :class:`~depz_camera_sdk.filters.base.Filter`,
:class:`~depz_camera_sdk.inference.base.DepthInferenceBase`,
:class:`~depz_camera_sdk.calibration.base.Calibration`,
:class:`~depz_camera_sdk.pipeline.depth_pipeline.DepthPipeline`.

A :class:`Control` is the self-description of one runtime parameter (for
UI + validation). The typed subclasses :class:`IntControl` / :class:`FloatControl`
/ :class:`BoolControl` / :class:`StrControl` additionally act as read-only
*descriptors* (via :class:`_ControlDescriptor`) — a host declares each control
as a class attribute::

    class SomeFilter(Filter):
        enabled  = BoolControl(default=False, label="Enable")
        strength = FloatControl("number", default=0.5,
                             min=0.0, max=1.0, step=0.05)

From that single declaration:
  - the attribute name (``strength``) becomes the control's ``name``;
  - ``obj.strength`` reads the current value, typed (``float`` here) —
    the IDE / type-checker resolve it because the descriptor's
    ``__get__`` is typed per subclass;
  - ``obj.strength = x`` raises — writes go through :meth:`ControlOwner.control_set`;
  - :meth:`ControlOwner._seed_control_defaults` populates it from ``default``.

A :class:`Control` carries two orthogonal axes, both explicit:

  - ``kind``       — *how the UI renders it* (slider / dropdown /
                     checkbox). ``BoolControl`` / ``StrControl`` default it.
  - ``value_type`` — *what the value is* (``int`` / ``float`` /
                     ``bool`` / ``str``). Fixed by the typed subclass.

Plain :class:`Control` (no descriptor behaviour) is also the data shape
reused by camera-control specs
(:class:`~depz_camera_sdk.cameras.camera_spec.CameraControl`), which
subclass it directly as data."""

import logging
import math
from dataclasses import asdict, dataclass
from typing import Any, ClassVar

_LOGGER = logging.getLogger("depz_camera_sdk.controls")

# Tolerance for "did coercion materially change the value?" checks.
# Float step-snapping (e.g. 0.05 grids) and int rounding accumulate
# tiny representation error; a difference within this tolerance is
# noise, not a real clamp/snap/round, and must not trigger a warning.
_COERCE_EPS_REL = 1e-9
_COERCE_EPS_ABS = 1e-12


def _coerce_changed(before: float, after: float) -> bool:
    """True iff ``after`` differs from ``before`` by more than float
    representation noise — i.e. coercion really moved the value."""
    return not math.isclose(before, after,
                            rel_tol=_COERCE_EPS_REL, abs_tol=_COERCE_EPS_ABS)


# Allowed ``Control.kind`` / ``Control.value_type`` strings — module
# constants so validation and any external caller check against the
# same source of truth.
CONTROL_KINDS: frozenset[str] = frozenset({"number", "enum", "boolean"})
CONTROL_VALUE_TYPES: frozenset[str] = frozenset({"int", "float", "bool", "str"})

# Which ``value_type``s each ``kind`` admits. A checkbox is always a
# bool; a slider is an int or a float; a dropdown can carry any of the
# three scalar types.
_KIND_VALUE_TYPES: "dict[str, frozenset[str]]" = {
    "boolean": frozenset({"bool"}),
    "number": frozenset({"int", "float"}),
    "enum": frozenset({"int", "float", "str"}),
}


@dataclass(frozen=True)
class Control:
    """Self-description of one runtime control — pure data (fields +
    validation + :meth:`to_dict`).

    Declare controls on a host via the typed subclasses (:class:`IntControl`
    etc.), which add read-only descriptor behaviour. ``Control`` itself is
    the shared base, and the plain data shape that camera-control specs
    (:class:`~depz_camera_sdk.cameras.camera_spec.CameraControl`)
    subclass directly."""

    kind: str
    """Widget hint: ``"number"`` / ``"enum"`` / ``"boolean"``. Controls
    UI rendering only — see ``value_type`` for the data side."""

    value_type: str
    """Data type — one of :data:`CONTROL_VALUE_TYPES`. Fixed by the typed
    subclass; :meth:`ControlOwner._coerce` keys off it."""

    default: Any = None
    """Factory default. Used by UI 'Reset' and to seed the backing
    attribute (:meth:`ControlOwner._seed_control_defaults`)."""

    label: str = ""
    """Human-readable label for the UI row. Empty → use ``name``."""

    min: float | None = None
    """Inclusive lower bound for ``kind="number"``."""

    max: float | None = None
    """Inclusive upper bound for ``kind="number"``."""

    step: float | None = None
    """Quantization for ``kind="number"`` sliders. Discrete-grid controls
    (e.g. block_size must be odd) declare ``step`` + ``min``/``max``;
    the base coercer enforces grid alignment."""

    enum_values: "tuple[tuple[int | float | str, str], ...]" = ()
    """``((value, label), ...)`` for ``kind="enum"`` only. The first
    slot is the value (numeric for ``value_type`` int/float, a string
    for ``value_type="str"``); the second is its UI label."""

    unit: str = ""
    """Unit suffix (``"px"``, ``"%"``, ``"m"``); empty when unitless."""

    name: str = ""
    """Control id within its owner. Filled automatically from the class
    attribute name when used as a control descriptor (see
    :class:`_ControlDescriptor`); passed explicitly when constructed as
    plain data (:class:`~depz_camera_sdk.cameras.camera_spec.CameraControl`)."""

    # ── Validation ────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        # Data use (CameraControl) passes ``name`` explicitly → validate
        # now. Descriptor use leaves ``name`` empty until __set_name__
        # (see _ControlDescriptor), which validates once the name is known.
        if self.name:
            self._validate()

    def _validate(self) -> None:
        """Fail-fast validation of this static declaration — unknown
        ``kind``, a ``value_type`` the ``kind`` can't carry, an enum
        without members, a ``default`` outside its own range. Raises at
        import, not on the first ``control_set`` in production."""
        if self.kind not in CONTROL_KINDS:
            raise ValueError(
                f"control {self.name!r}: kind must be one of "
                f"{sorted(CONTROL_KINDS)}, got {self.kind!r}")
        if self.value_type not in CONTROL_VALUE_TYPES:
            raise ValueError(
                f"control {self.name!r}: value_type must be one of "
                f"{sorted(CONTROL_VALUE_TYPES)}, got {self.value_type!r}")
        allowed_vt = _KIND_VALUE_TYPES[self.kind]
        if self.value_type not in allowed_vt:
            raise ValueError(
                f"control {self.name!r}: kind={self.kind!r} admits "
                f"value_type {sorted(allowed_vt)}, got {self.value_type!r}")
        if self.kind == "enum":
            self._validate_enum()
        elif self.enum_values:
            raise ValueError(
                f"control {self.name!r}: enum_values is only valid for "
                f"kind='enum', not kind={self.kind!r}")
        if self.kind == "number":
            self._validate_number()

    def _validate_enum(self) -> None:
        """``kind="enum"``: members must be non-empty, type-match
        ``value_type``, and the ``default`` (if set) must be one of
        them."""
        if not self.enum_values:
            raise ValueError(
                f"control {self.name!r}: kind='enum' requires a non-empty "
                f"enum_values")
        members = [v for v, _ in self.enum_values]
        if self.value_type == "str":
            bad = [v for v in members if not isinstance(v, str)]
            if bad:
                raise ValueError(
                    f"control {self.name!r}: value_type='str' enum requires "
                    f"string members; got non-strings {bad}")
            allowed: set = set(members)
        else:  # int / float
            try:
                nums = [float(v) for v in members]
            except (TypeError, ValueError):
                raise ValueError(
                    f"control {self.name!r}: numeric enum has non-numeric "
                    f"members in {members}")
            if self.value_type == "int" and not all(
                    n.is_integer() for n in nums):
                raise ValueError(
                    f"control {self.name!r}: value_type='int' enum requires "
                    f"integral members; got {members}")
            allowed = set(nums)
        if self.default is not None:
            probe = (self.default if self.value_type == "str"
                     else float(self.default))
            if probe not in allowed:
                raise ValueError(
                    f"control {self.name!r}: default {self.default!r} is not "
                    f"one of the enum values {sorted(allowed)}")

    def _validate_number(self) -> None:
        """``kind="number"``: ``min``/``max``/``step`` must be sane and
        ``default`` (if set) must fall inside ``[min, max]``."""
        if (self.min is not None and self.max is not None
                and self.min > self.max):
            raise ValueError(
                f"control {self.name!r}: min ({self.min}) > max ({self.max})")
        if self.step is not None and self.step <= 0:
            raise ValueError(
                f"control {self.name!r}: step must be positive, got {self.step}")
        if self.default is not None:
            d = float(self.default)
            if self.min is not None and d < self.min:
                raise ValueError(
                    f"control {self.name!r}: default {self.default!r} is below "
                    f"min {self.min}")
            if self.max is not None and d > self.max:
                raise ValueError(
                    f"control {self.name!r}: default {self.default!r} is above "
                    f"max {self.max}")

    def to_dict(self) -> dict:
        return asdict(self)


# ── Control descriptor ───────────────────────────────────────────────────

@dataclass(frozen=True)
class _ControlDescriptor(Control):
    """A :class:`Control` that also acts as a read-only descriptor — this
    is what a :class:`ControlOwner` host declares as a class attribute.

    The split keeps the two roles honest: plain :class:`Control` is the
    data shape (reused by camera-control specs), and only this branch
    carries the descriptor protocol. Use the typed subclasses
    (:class:`IntControl` etc.), never ``_ControlDescriptor`` directly."""

    def __set_name__(self, owner: type, name: str) -> None:
        # `Control` is frozen — use object.__setattr__ to record the
        # attribute name the host bound us to, then validate now that
        # the name (used in error messages) is known.
        object.__setattr__(self, "name", name)
        self._validate()

    def __get__(self, obj: "object | None", objtype: "type | None" = None) -> Any:
        """Read the live value from the host's ``_control_<name>`` backing
        attribute. Class access (``Host.control``) returns the descriptor
        itself so the catalog machinery can introspect it."""
        if obj is None:
            return self
        return getattr(obj, f"_control_{self.name}")

    def __set__(self, obj: object, value: Any) -> None:
        raise AttributeError(
            f"control {self.name!r} is read-only — write it via "
            f"control_set({self.name!r}, ...)")


# ── Typed control subclasses ─────────────────────────────────────────────
#
# A host declares each control with one of these. The subclass fixes
# ``value_type`` and the ``__get__`` return type, so reading
# ``host.<control>`` resolves to a concrete type in the IDE / type-checker.
# ``__get__`` delegates to the base (typed ``Any``), so narrowing the
# return annotation here is type-safe.

@dataclass(frozen=True)
class IntControl(_ControlDescriptor):
    """A control whose value is an ``int`` (``"number"`` slider or
    ``"enum"`` dropdown of integer codes)."""
    value_type: str = "int"

    def __get__(self, obj: "object | None",
                objtype: "type | None" = None) -> int:
        return super().__get__(obj, objtype)


@dataclass(frozen=True)
class FloatControl(_ControlDescriptor):
    """A control whose value is a ``float`` (``"number"`` slider)."""
    value_type: str = "float"

    def __get__(self, obj: "object | None",
                objtype: "type | None" = None) -> float:
        return super().__get__(obj, objtype)


@dataclass(frozen=True)
class BoolControl(_ControlDescriptor):
    """A control whose value is a ``bool`` (a checkbox). ``kind`` defaults
    to ``"boolean"``."""
    kind: str = "boolean"
    value_type: str = "bool"

    def __get__(self, obj: "object | None",
                objtype: "type | None" = None) -> bool:
        return super().__get__(obj, objtype)


@dataclass(frozen=True)
class StrControl(_ControlDescriptor):
    """A control whose value is a ``str``. ``kind`` defaults to ``"enum"``
    (string controls are always dropdowns of a fixed value set)."""
    kind: str = "enum"
    value_type: str = "str"

    def __get__(self, obj: "object | None",
                objtype: "type | None" = None) -> str:
        return super().__get__(obj, objtype)


class ControlOwner:
    """Mixin: control storage, accessors, and the coerce-clamp-snap logic.

    A host declares its controls as class attributes (typed
    :class:`IntControl` / :class:`FloatControl` / :class:`BoolControl` /
    :class:`StrControl` descriptors). :meth:`__init_subclass__` collects
    them — across the MRO, so controls inherited from a base class compose
    with the subclass's own — into :data:`_CONTROL_SPECS` (ordered tuple)
    and :data:`_control_index` (name → :class:`Control`).

    The host calls :meth:`_seed_control_defaults` once in ``__init__`` to
    populate the ``_control_<name>`` backing attributes from each
    descriptor's ``default`` — the default lives in exactly one place.

    The rest is inherited:
      - reading ``host.<control>`` goes through the descriptor (typed,
        read-only); ``control_get(name)`` is the by-string equivalent;
      - :meth:`control_set(name, value)` → coerce + store. Subclasses that
        need side effects (matcher rebuilds, invariants) override and
        call ``super().control_set(...)`` first;
      - :meth:`controls_to_dict()` → snapshot every current value.

    The ``control_`` prefix on the accessors is intentional: hosts
    (``DepthPipeline``, ``Calibration``) already own ``get`` / ``set`` /
    ``to_dict`` for their own primary APIs."""

    # Populated per class by __init_subclass__ from the Control descriptors
    # declared (or inherited) on the class.
    _CONTROL_SPECS: ClassVar["tuple[Control, ...]"] = ()
    _control_index: ClassVar["dict[str, Control]"] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Collect every Control descriptor across the MRO — base-defined
        # controls first, subclass overrides win, class-body order kept
        # (dict preserves insertion order; re-insert keeps position).
        collected: "dict[str, Control]" = {}
        for klass in reversed(cls.__mro__):
            for attr, val in vars(klass).items():
                if isinstance(val, _ControlDescriptor):
                    collected[attr] = val
        cls._control_index = collected
        cls._CONTROL_SPECS = tuple(collected.values())

    # ── Public API ───────────────────────────────────────────────────

    def control_get(self, name: str) -> Any:
        """Return the current value of one control by name — reads its
        ``_control_<name>`` backing attribute. Equivalent to ``host.<name>``
        (the descriptor) but takes the name as a string."""
        return getattr(self, self._control_attr(self._resolve_control(name)))

    def control_set(self, name: str, value: Any) -> Any:
        """Apply one control update: look up descriptor, coerce + clamp by
        ``kind``/``value_type``/``min``/``max``/``step``, ``setattr``
        onto ``_control_<name>``. Returns the value actually stored.

        Subclasses override to add side effects (e.g. SGBM rebuilds
        its matcher when controls change) — call ``super().control_set(...)``
        for the coerce/store, then act on the new value."""
        control = self._resolve_control(name)
        coerced = self._coerce(control, value)
        setattr(self, self._control_attr(control), coerced)
        return coerced

    @staticmethod
    def _control_attr(control: Control) -> str:
        """Backing-attribute name for ``control`` — always ``_control_<name>``.
        The ``_control_`` prefix namespaces control storage away from a host's
        other private state (``_matcher``, ``_prev``, buffers, ...)."""
        return f"_control_{control.name}"

    def _seed_control_defaults(self) -> None:
        """Populate every ``_control_<name>`` backing attribute from its
        descriptor's ``default``. Hosts call this once, early in
        ``__init__`` — before any ``control_set`` / rebuild hook. The
        default lives only in :attr:`Control.default`; hosts repeat it
        nowhere."""
        for k in self.controls():
            self._store_control_raw(k.name, k.default)

    def _store_control_raw(self, name: str, value: Any) -> None:
        """Write a control's ``_control_<name>`` backing attribute directly —
        **no coercion, no side effects**. The single named escape hatch
        for the handful of in-class cases that legitimately bypass
        :meth:`control_set`:

          - :meth:`_seed_control_defaults` populating descriptor defaults;
          - a ctor applying a constructor argument before the object is
            fully built (a side-effect hook like a matcher rebuild
            would fire on a half-constructed object);
          - a sentinel value deliberately outside the descriptor's
            range (``rectify_alpha = -1`` 'auto');
          - an invariant correction inside a :meth:`control_set` override
            (ZClip's ``max > min``).

        ``name`` is still validated against the control index. Application
        code must never call this — use :meth:`control_set`."""
        setattr(self, self._control_attr(self._resolve_control(name)), value)

    def controls_to_dict(self) -> dict:
        """Snapshot every current control value — for the UI live-state
        pane, recorder sidecar, etc."""
        return {k.name: self.control_get(k.name) for k in self.controls()}

    def controls(self) -> "tuple[Control, ...]":
        """This host's control descriptors — the class-level
        :data:`_CONTROL_SPECS` tuple (same object every call, no per-call
        allocation). The single public accessor shared by every
        ControlOwner: filters, inference, calibration, pipeline."""
        return type(self)._CONTROL_SPECS

    # ── Internals ────────────────────────────────────────────────────

    def _resolve_control(self, name: str) -> Control:
        control = type(self)._control_index.get(name)
        if control is None:
            owner = getattr(self, "SLUG", "") or type(self).__name__
            raise KeyError(
                f"unknown control {name!r} for {owner!r}; "
                f"valid: {list(type(self)._control_index)}")
        return control

    # ── Coercion ──────────────────────────────────────────────────────
    #
    # ``_coerce`` is a plain dispatch on ``kind``; each branch is its
    # own small, independently testable helper. The result type is
    # decided solely by ``control.value_type``.

    @staticmethod
    def _coerce(control: Control, value: Any) -> Any:
        """Coerce ``value`` for ``control`` — dispatch by ``kind``."""
        if control.kind == "boolean":
            return bool(value)
        if control.kind == "enum":
            return ControlOwner._coerce_enum(control, value)
        if control.kind == "number":
            return ControlOwner._coerce_number(control, value)
        raise ValueError(f"unknown control kind {control.kind!r}")

    @staticmethod
    def _coerce_number(control: Control, value: Any) -> "int | float":
        """``kind="number"``: cast to float, clamp to ``[min, max]``,
        snap to the ``step`` grid, then return ``int`` or ``float``
        per ``control.value_type``.

        Each transform that *materially* changes the value (clamp to a
        bound, snap to the step grid, round a fractional float to int)
        emits a ``WARNING`` — a silent clamp otherwise hides a caller
        bug (passed 1000, got 64, no signal). Float-noise-sized
        differences are not warned about (see :func:`_coerce_changed`)."""
        raw = float(value)
        v = ControlOwner._clamp(raw, control.min, control.max)
        if _coerce_changed(raw, v):
            _LOGGER.warning(
                "control %r: value %r out of range [%s, %s] — clamped to %r",
                control.name, raw, control.min, control.max, v)
        if control.step is not None and control.step > 0:
            snapped = ControlOwner._snap_to_grid(v, control.step, control.min, control.max)
            if _coerce_changed(v, snapped):
                _LOGGER.warning(
                    "control %r: value %r off the %s-step grid — snapped to %r",
                    control.name, v, control.step, snapped)
            v = snapped
        if control.value_type == "int":
            iv = int(round(v))
            if _coerce_changed(v, iv):
                _LOGGER.warning(
                    "control %r: value %r is not integral — rounded to %d",
                    control.name, v, iv)
            return iv
        return float(v)

    @staticmethod
    def _coerce_enum(control: Control, value: Any) -> Any:
        """``kind="enum"``: ``value`` must match one of ``enum_values``.
        ``value_type="str"`` matches by string equality; numeric enums
        compare as floats and return ``int``/``float`` per
        ``value_type``."""
        allowed = [val for val, _ in control.enum_values]
        if control.value_type == "str":
            if value not in allowed:
                raise ValueError(
                    f"value {value!r} not in enum {allowed} "
                    f"for {control.name!r}")
            return value
        v = float(value)
        allowed_set = {float(a) for a in allowed}
        if v not in allowed_set:
            raise ValueError(
                f"value {value!r} not in enum {sorted(allowed_set)} "
                f"for {control.name!r}")
        return int(v) if control.value_type == "int" else v

    @staticmethod
    def _clamp(v: float, lo: "float | None", hi: "float | None") -> float:
        """Clamp ``v`` into the inclusive ``[lo, hi]`` range; either
        bound may be ``None`` (open on that side)."""
        if lo is not None:
            v = max(lo, v)
        if hi is not None:
            v = min(hi, v)
        return v

    @staticmethod
    def _snap_to_grid(v: float, step: float,
                      lo: "float | None", hi: "float | None") -> float:
        """Snap ``v`` to the nearest ``lo + n*step`` grid point (grid
        origin is ``lo``, or ``0.0`` when ``lo`` is open), then re-clamp
        to ``hi`` so rounding can't push past the upper bound."""
        base = lo if lo is not None else 0.0
        v = base + round((v - base) / step) * step
        return min(hi, v) if hi is not None else v
