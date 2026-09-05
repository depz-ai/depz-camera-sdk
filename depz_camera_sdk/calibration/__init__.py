"""Stereo calibration — ABC + registry + concrete impls + resolver.

Public exports:
- :class:`Calibration` — abstract base
- :class:`CanvasLayout` — geometry of rectified eye inside the canvas
- :class:`OpenCVBasicCalibration` — pinhole + radtan8 (cv2.stereoRectify)
- :class:`OpenCVFisheyeCalibration` — Kannala–Brandt (cv2.fisheye)
- :func:`register_calibration` — decorator for adding new kinds
- :func:`registered_polymorphic_names` — list registered cereal lens-model names
"""

from .base import (
    Calibration,
    CanvasLayout,
    register_calibration,
    registered_polymorphic_names,
)
from .opencv_basic import OpenCVBasicCalibration
from .opencv_fisheye import OpenCVFisheyeCalibration

__all__ = [
    "Calibration",
    "CanvasLayout",
    "OpenCVBasicCalibration",
    "OpenCVFisheyeCalibration",
    "register_calibration",
    "registered_polymorphic_names",
]
