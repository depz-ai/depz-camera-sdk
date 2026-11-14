"""What the CUDA 12 and CUDA 13 lines of ONNX Runtime's CUDA execution
provider need from the host: runtime library file names, where NVIDIA's
wheels put them, and the driver / GPU floors.

Two lines are supported side by side, one per environment:

* **CUDA 12** — ``onnxruntime-gpu`` 1.19–1.26, the ``cuda12`` /
  ``cuda12-bundled`` extras. Runs on every GPU the CUDA EP supports.
* **CUDA 13** — ``onnxruntime-gpu`` 1.27+, the ``cuda13`` /
  ``cuda13-bundled`` extras. Turing-or-newer GPUs, driver 580+.

Which line an environment is on is decided by the installed onnxruntime
wheel alone — :func:`installed_cuda_major` reads it back, so nothing else
has to be configured or kept in sync.

This is the one place those facts are written down for installed code:
the SDK's Windows DLL pre-load (:mod:`.windows`), the viewer's "is a
system CUDA runtime present?" probe and its driver-trouble checks all read
them from here, so a CUDA or cuDNN version bump is a one-file change.

``installer/bootstrap_core.py`` carries its own copy of the probe-library
and driver-floor tables — it runs before the SDK is installed and cannot
import this module. Tests keep the copies equal.
"""

from __future__ import annotations

import functools
import importlib.util
import re
from pathlib import Path
from typing import Optional

# CUDA majors with an extras pair, oldest first.
SUPPORTED_CUDA_MAJORS: tuple[int, ...] = (12, 13)

# What "NVIDIA" means when nothing says otherwise: auto-detect installs it
# and it is the answer for an environment whose line cannot be read.
DEFAULT_CUDA_MAJOR = 12

# The two libraries whose loadability answers "can the CUDA EP start
# here?" — cuBLAS-Lt is the first import the provider DLL fails on, and
# cuDNN is the piece a plain CUDA Toolkit install does not include.
# Both lines use cuDNN 9. Keyed by CUDA major, then the viewer's OS names.
CUDA_SYSTEM_PROBE_LIBS: dict[int, dict[str, tuple[str, ...]]] = {
    12: {"windows": ("cublasLt64_12.dll", "cudnn64_9.dll"),
         "linux":   ("libcublasLt.so.12", "libcudnn.so.9")},
    13: {"windows": ("cublasLt64_13.dll", "cudnn64_9.dll"),
         "linux":   ("libcublasLt.so.13", "libcudnn.so.9")},
}

# Everything the CUDA EP needs on Windows, in the order it must be
# loaded; each comes from the NVIDIA wheel noted.
#
# nvJitLink must load before cuBLAS — cuBLAS imports it. Note the wheel
# ships it as `nvJitLink_1X0_0.dll` (not the `*64_1X` soversion naming the
# other libs use), so load it by that exact name first; otherwise cuBLAS's
# transitive dependency resolution can miss it and the whole CUDA EP fails
# with a misleading "cublasLt64_1X.dll missing".
#
# The library versions inside one toolkit are NOT all the toolkit's major:
# cuFFT is 11 in CUDA 12 and 12 in CUDA 13, cuRAND is 10 in both.
CUDA_WINDOWS_PRELOAD_DLLS: dict[int, tuple[str, ...]] = {
    12: (
        "cudart64_12.dll",      # nvidia-cuda-runtime-cu12
        "nvJitLink_120_0.dll",  # nvidia-nvjitlink-cu12
        "cublas64_12.dll",      # nvidia-cublas-cu12
        "cublasLt64_12.dll",    # nvidia-cublas-cu12 — the one that fails first
        "cufft64_11.dll",       # nvidia-cufft-cu12
        "curand64_10.dll",      # nvidia-curand-cu12
        "cudnn64_9.dll",        # nvidia-cudnn-cu12
        "nvrtc64_120_0.dll",    # nvidia-cuda-nvrtc-cu12
    ),
    13: (
        "cudart64_13.dll",      # nvidia-cuda-runtime
        "nvJitLink_130_0.dll",  # nvidia-nvjitlink
        "cublas64_13.dll",      # nvidia-cublas
        "cublasLt64_13.dll",    # nvidia-cublas — the one that fails first
        "cufft64_12.dll",       # nvidia-cufft
        "curand64_10.dll",      # nvidia-curand
        "cudnn64_9.dll",        # nvidia-cudnn-cu13
        "nvrtc64_130_0.dll",    # nvidia-cuda-nvrtc
    ),
}

# Oldest NVIDIA driver each line's runtime accepts, as (major, minor).
# CUDA 12: the 12.0 GA floors. CUDA 13: NVIDIA documents ">= 580" for the
# whole 13.x family and 580.65.06 as the Linux driver of 13.0 GA; it
# publishes no separate Windows number, so the branch start is the floor.
CUDA_MIN_DRIVER: dict[int, dict[str, tuple[int, int]]] = {
    12: {"linux": (525, 60), "windows": (527, 41)},
    13: {"linux": (580, 65), "windows": (580, 0)},
}

# Oldest GPU compute capability each line runs on. CUDA 13 removed library
# support for Maxwell (5.x), Pascal (6.x) and Volta (7.0): Turing (7.5) is
# the floor. The CUDA 12 figure is the ORT CUDA EP's own floor (Pascal).
CUDA_MIN_COMPUTE_CAPABILITY: dict[int, tuple[int, int]] = {
    12: (6, 0),
    13: (7, 5),
}

# First onnxruntime-gpu minor built against each CUDA major on PyPI —
# the fallback when the wheel's build info does not state its CUDA version.
_FIRST_ORT_MINOR: tuple[tuple[int, int], ...] = ((27, 13), (19, 12))


def _onnxruntime_dir() -> Optional[Path]:
    """Directory of the installed ``onnxruntime`` package, WITHOUT
    importing it (the import loads native code; on Windows that must not
    happen before the CUDA DLLs are pre-loaded)."""
    try:
        spec = importlib.util.find_spec("onnxruntime")
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    if spec.origin:
        return Path(spec.origin).parent
    for loc in spec.submodule_search_locations or ():
        return Path(loc)
    return None


def parse_cuda_major(build_info: str) -> Optional[int]:
    """CUDA major from the text of onnxruntime's
    ``capi/build_and_package_info.py`` (``cuda_version = '12.8'``), or
    None for a wheel that is not a CUDA build."""
    m = re.search(r"""^cuda_version\s*=\s*['"](\d+)""", build_info, re.M)
    if m:
        return int(m.group(1))
    # No cuda_version line. Only onnxruntime-gpu is a CUDA build; derive
    # its line from the release number.
    if not re.search(r"""^package_name\s*=\s*['"]onnxruntime-gpu['"]""",
                     build_info, re.M):
        return None
    v = re.search(r"""^__version__\s*=\s*['"](\d+)\.(\d+)""", build_info, re.M)
    if not v or int(v.group(1)) != 1:
        return None
    for first_minor, major in _FIRST_ORT_MINOR:
        if int(v.group(2)) >= first_minor:
            return major
    return None


@functools.lru_cache(maxsize=1)
def installed_cuda_major() -> Optional[int]:
    """CUDA major the installed onnxruntime was built against (12 or 13),
    or None when no CUDA build of onnxruntime is installed."""
    pkg = _onnxruntime_dir()
    if pkg is None:
        return None
    try:
        text = (pkg / "capi" / "build_and_package_info.py").read_text(
            encoding="utf-8", errors="ignore")
    except OSError:
        return None
    return parse_cuda_major(text)


def cuda_major_or_default() -> int:
    """:func:`installed_cuda_major`, falling back to the default line —
    for callers that need *a* table row even on a non-CUDA install."""
    major = installed_cuda_major()
    return major if major in SUPPORTED_CUDA_MAJORS else DEFAULT_CUDA_MAJOR


def suffixless_wheel_majors(nvidia_root: Path) -> list[int]:
    """CUDA majors whose suffix-less runtime wheels (the ``nvidia/cuNN/``
    layout NVIDIA introduced with CUDA 13) are present under one
    ``site-packages/nvidia``, ascending."""
    if not nvidia_root.is_dir():
        return []
    found = []
    for d in nvidia_root.iterdir():
        m = re.fullmatch(r"cu(\d+)", d.name)
        if m and d.is_dir():
            found.append(int(m.group(1)))
    return sorted(found)


def windows_bin_dirs(nvidia_root: Path, major: int) -> list[Path]:
    """Directories under one ``site-packages/nvidia`` that hold the DLLs
    of CUDA line ``major``.

    The two generations of NVIDIA wheels lay their DLLs out differently::

        CUDA 12   nvidia/<lib>/bin/                (cublas, cudnn, cufft, …)
        CUDA 13   nvidia/cu13/bin/x86_64/          (everything but cuDNN)
                  nvidia/cudnn/bin/                (cuDNN, unchanged)

    Only the line's own directories are returned, so an environment that
    still carries the other line's leftovers cannot pre-load the wrong
    cuBLAS."""
    if not nvidia_root.is_dir():
        return []
    own = nvidia_root / f"cu{major}"
    if own.is_dir():
        # Suffix-less generation: its own dir plus cuDNN's.
        dirs = [d for d in (own / "bin" / "x86_64", own / "bin") if d.is_dir()][:1]
        cudnn = nvidia_root / "cudnn" / "bin"
        if cudnn.is_dir():
            dirs.append(cudnn)
        return dirs
    # Per-library generation: nvidia/<lib>/bin, skipping any cuNN tree.
    return [lib_dir / "bin" for lib_dir in sorted(nvidia_root.iterdir())
            if not re.fullmatch(r"cu\d+", lib_dir.name)
            and (lib_dir / "bin").is_dir()]
