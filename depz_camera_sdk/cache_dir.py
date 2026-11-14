"""SDK-wide cache directory.

The SDK and components built on it persist on-disk state across runs
under one root: downloaded models, the engines' compiled-model caches
(which turn a 30–60 s GPU compile into a <1 s load), the viewer's
camera-info store.

Resolution order for :func:`get_cache_dir`:

  1. A :func:`set_cache_dir` override for the current process.
  2. The platform's per-user cache location (:func:`default_cache_dir`):

     - Linux:   ``$XDG_CACHE_HOME/depz``, else ``~/.cache/depz``
     - macOS:   ``~/Library/Caches/depz``
     - Windows: ``%LOCALAPPDATA%\\depz\\Cache``

The default never depends on the current working directory: the same
cache is found wherever a script is started from, and a read-only
working directory is not a problem.

Consumers subdivide the root by purpose:

    cache_root = get_cache_dir()
    models = cache_root / "models"
    openvino_cache = cache_root / "openvino"
    camera_info_cache = cache_root / "camera-info"
"""

import os
import sys
from pathlib import Path
from typing import Mapping, Optional, Union


_OVERRIDE: Optional[Path] = None


def _platform_cache_dir(platform: str, env: Mapping[str, str],
                        home: Path) -> Path:
    """Per-user cache root for ``platform`` (a ``sys.platform`` value).
    Pure — everything it depends on is passed in."""
    if platform == "darwin":
        return home / "Library" / "Caches" / "depz"
    if platform == "win32":
        local_app_data = env.get("LOCALAPPDATA")
        base = Path(local_app_data) if local_app_data else (
            home / "AppData" / "Local")
        return base / "depz" / "Cache"
    # Linux and everything else that follows the XDG base-directory
    # spec, which says a relative XDG_CACHE_HOME must be ignored.
    xdg = env.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg and os.path.isabs(xdg) else home / ".cache"
    return base / "depz"


def default_cache_dir() -> Path:
    """The per-user cache root the SDK uses when no override is set —
    see the module docstring for the per-platform locations."""
    return _platform_cache_dir(sys.platform, os.environ, Path.home())


def get_cache_dir() -> Path:
    """Return the active SDK cache root (always an absolute path): the
    :func:`set_cache_dir` override if one is set, else
    :func:`default_cache_dir`. The directory is not created here —
    whoever writes into it creates it."""
    if _OVERRIDE is not None:
        return _OVERRIDE
    return default_cache_dir()


def set_cache_dir(path: Union[str, Path]) -> None:
    """Override the cache root for the current process, e.g. to keep
    everything on a data volume. ``~`` and environment variables in
    ``path`` are expanded, and a relative path is made absolute against
    the working directory at the time of the call — later ``chdir``\\ s
    do not move the cache. Child processes do not inherit the override."""
    global _OVERRIDE
    expanded = os.path.expandvars(os.path.expanduser(os.fspath(path)))
    _OVERRIDE = Path(os.path.abspath(expanded))


def clear_cache_dir_override() -> None:
    """Drop any :func:`set_cache_dir` override; :func:`get_cache_dir`
    returns :func:`default_cache_dir` again. Mostly for tests."""
    global _OVERRIDE
    _OVERRIDE = None
