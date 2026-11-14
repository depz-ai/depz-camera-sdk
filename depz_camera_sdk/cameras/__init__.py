"""Device registry + device-info endpoint client + camera discovery.

This package owns everything camera-shaped:
- :class:`CameraSpec` — firmware-class description (capture modes,
  controls, SKUs).
- Per-class registrations (Varius, Minima, Clarity, Mono, Mono Mini).
- :func:`fetch_camera_info_json` — server's device-info endpoint.
- :func:`list_connected_depz_cameras` /
  :func:`get_first_connected_depz_camera` /
  :func:`get_connected_depz_camera_by_serial` — connected-device discovery
  (V4L2/DShow → SKU → CameraSpec).
"""

from .web_api import fetch_camera_info_json
from .camera_info import (
    DepzCameraInfo,
    get_first_connected_depz_camera,
    get_connected_depz_camera_by_serial,
    list_connected_depz_cameras,
    list_connected_depz_cameras_detailed,
)
from .depz_clarity import DEPZ_CLARITY
from .depz_minima import DEPZ_MINIMA
from .depz_mono import DEPZ_MONO
from .depz_mono_mini import DEPZ_MONO_MINI
from .depz_varius import DEPZ_VARIUS
from .camera_spec import REGISTRY, spec_by_sku
from .camera_spec import CameraSpec

__all__ = [
    "CameraSpec",
    "DEPZ_VARIUS", "DEPZ_MINIMA", "DEPZ_CLARITY",
    "DEPZ_MONO", "DEPZ_MONO_MINI",
    "REGISTRY", "spec_by_sku",
    "fetch_camera_info_json",
    "DepzCameraInfo", "list_connected_depz_cameras",
    "list_connected_depz_cameras_detailed",
    "get_first_connected_depz_camera",
    "get_connected_depz_camera_by_serial",
]
