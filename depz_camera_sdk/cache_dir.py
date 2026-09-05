"""SDK-wide cache directory.

The SDK and components built on it (inference engines, web viewer's
camera-info store, future ones) may need to persist on-disk state
across runs — most importantly OpenVINO's compiled-model cache,
which turns 30–60 s GPU compiles into <1 s loads on subsequent
starts.

Resolution order for :func:`get_cache_dir`:

  1. :func:`set_cache_dir` runtime override (used by hosting apps
     like the web viewer to redirect into a per-platform user-cache
     location).
  2. ``./depz_sdk_cache/`` in the current working directory.

The CWD-based default is intentionally simple — it keeps cache state
local to the project the user is running from, no surprise pollution
of ``$HOME``. Apps that want OS-standard locations
(``~/.cache/depz/`` on Linux / ``~/Library/Caches/depz/`` on macOS
/ ``%LOCALAPPDATA%\\depz\\Cache\\`` on Windows) call
:func:`set_cache_dir` at startup.

Engines and other consumers subdivide the cache by purpose:

    cache_root = get_cache_dir()
    openvino_cache = cache_root / "openvino"
    camera_info_cache = cache_root / "camera-info"
"""

from pathlib import Path
from typing import Optional, Union


_OVERRIDE: Optional[Path] = None


def get_cache_dir() -> Path:
    """Return the active SDK cache root. See module docstring for
    resolution order. Caller is responsible for creating the directory
    if it doesn't exist (it's lazy)."""
    if _OVERRIDE is not None:
        return _OVERRIDE
    return Path.cwd() / "depz_sdk_cache"


def set_cache_dir(path: Union[str, Path]) -> None:
    """Override the SDK-default cache root for the current process.
    Hosting apps (e.g. the web viewer) call this at startup to redirect
    cache into a platform-standard application data dir. Pass an
    absolute path; ``~``-prefix expanded automatically."""
    global _OVERRIDE
    _OVERRIDE = Path(path).expanduser()


def clear_cache_dir_override() -> None:
    """Drop any :func:`set_cache_dir` override; subsequent calls fall
    back to the CWD default. Mostly for tests."""
    global _OVERRIDE
    _OVERRIDE = None
