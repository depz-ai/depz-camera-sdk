"""Source-side value types — produced by :class:`FrameSource` and
consumed by the rest of the SDK.

:class:`RawFrame` is what a source's ``grab()`` returns each iteration
(packet bytes + timestamp). :class:`CameraInfo` is what ``open()``
returns — the geometry + identity descriptor of the configured stream.
Both live here because sources are the producers; the pipeline
imports these types downstream from sources, which keeps the
dependency direction clean.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class RawFrame:
    """Raw frame straight from a FrameSource — packet bytes + a single
    timestamp.

    ``payload`` is the source's wire bytes — MJPEG packet for MJPG
    captures, raw YUYV frame for YUYV captures. Suitable for either
    decode (cv2.imdecode for MJPG) or fan-out to recorder (no
    re-encode).

    ``pts_us`` is microseconds since stream start on the source's clock
    domain. Source-local — integer math, no rounding. Live cameras
    anchor at the V4L2 timestamp of the first emitted frame; replay
    anchors at the MKV stream start (== container PTS).

    Absolute time on the source's clock domain is derived as
    ``source.t0_monotonic_ns + pts_us * 1000`` (returns int_ns,
    bit-exact, no float in the timing chain). For UTC use
    ``source.t0_utc_ns + pts_us * 1000``. The clock domain itself is
    on :class:`CameraInfo`, not duplicated per-frame.
    """
    payload: bytes
    """Raw wire bytes — MJPEG packet (or YUYV frame on YUYV captures).
    Suitable for ``cv2.imdecode`` or fan-out to recorder without
    re-encode."""

    pts_us: int
    """Microseconds since stream start on the source's clock domain.
    Source-local — integer math, no rounding. Combine with
    ``source.t0_monotonic_ns`` for absolute time."""

    sequence: int
    """Monotonic counter incremented per ``grab()``. Useful for debugging
    gaps; not the same as ``source_frame_idx``."""

    source_frame_idx: Optional[int] = None
    """Replay-only: index in the recording's frame index, set by
    ``MkvSource``. ``None`` on live."""


@dataclass(frozen=True, slots=True)
class CameraInfo:
    """Geometry + identity descriptor for a live or replayed source."""
    os_id: str
    """OS-level identifier for the camera — V4L2 path (``/dev/videoN``)
    on Linux, DirectShow display name on Windows, MKV file path on
    replay. Backend-specific string; not a path on every platform."""

    camera_name: str
    """Vendor/product string (kernel-reported on live, recorder-stamped
    on replay)."""

    camera_serial: str
    """DEPZ camera serial. Empty string if unrecognized."""

    width: int
    """Frame width as delivered by the source — full SBS for stereo
    captures (both eyes), single-eye width for mono captures. Use
    :attr:`CaptureMode.eye_w` if you need the per-eye width regardless
    of family."""

    height: int
    """Frame height in pixels (SBS doesn't split height, so this is
    the per-eye height for both stereo and mono)."""

    fps: float
    """Declared capture rate. May be fractional (V4L2 advertises
    7.5, 12.5, ...)."""

    is_live: bool
    """True for `CameraSource`; False for `MkvSource`. Drives UI flow
    gating."""

    clock_domain: str
    """`"linux_monotonic"` / `"windows_qpc"` / `"macos_mach"` — the
    reference clock for `pts_us`."""

    sku: Optional[str] = None
    """Camera SKU returned by the camera-info endpoint (e.g.
    ``"STEREO-CAMERA-VARIUS-REGULAR-BASELINE-REGULAR-LENSES-V21"``) when the
    source is a recognized DEPZ camera; ``None`` otherwise. Use
    :func:`spec_by_sku` to walk to the matched :class:`CameraSpec`."""
