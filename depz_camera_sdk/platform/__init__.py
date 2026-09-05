"""Platform abstraction.

Two real backends today: :class:`LinuxBackend` (V4L2 ioctl + sysfs
USB iProduct) and :class:`WindowsBackend` (WMI enumeration + DirectShow
``IAMVideoProcAmp``/``IAMCameraControl``). macOS is stubbed.

Concrete impl is selected by :func:`current_backend` and cached for
the process lifetime.
"""

import os as _os
import sys

# Must be a bare module-level statement so it fires before any
# cv2-importing submodule is reached. See platform/windows.py for why.
_os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")
del _os

from .base import (
    CameraControls,
    PlatformBackend,
    VideoDeviceInfo,
    _bootstrap,
    current_backend,
)

# Platform-specific CUDA runtime setup — runs once at import time,
# before the OpenCL toggle below so the NVIDIA ICD is already visible.
# darwin: no CUDA, no preload needed.
_bootstrap()
del _bootstrap

# Runtime OpenCL/UMat disable.
try:
    import cv2 as _cv2
    _cv2.ocl.setUseOpenCL(False)
    del _cv2
except Exception:
    pass


__all__ = ["PlatformBackend", "CameraControls", "VideoDeviceInfo", "current_backend"]
