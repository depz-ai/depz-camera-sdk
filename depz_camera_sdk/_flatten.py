"""Flattened multi-module extensions — N compiled modules in ONE .so/.pyd.

CPython binds a dynamic library to a module through its ``PyInit_<tail>``
symbol, one symbol per module — but nothing stops one shared object from
exporting SEVERAL init symbols. ``setup.py`` builds such fat extensions from
the groups declared in ``_compile_manifest.FLATTEN_GROUPS`` (build-only,
never installed); this module is the RUNTIME half: :func:`install` puts a
``sys.meta_path`` finder in front of the path-based finder so member imports
resolve from the fat file (``ExtensionFileLoader`` derives ``PyInit_<tail>``
from the module name's last segment). When no fat file exists — pure-Python
checkout, ``DEPZ_DEV_NO_COMPILE`` build — the finder maps nothing and plain
``.py`` members import exactly as before.

Membership is stored as SALTED DIGESTS of the member module names, not
plaintext: this file ships as ``.py`` (it must be importable before any
extension loads) and there is no reason for it to document the compiled
module layout. Real names exist only in the build-only manifest and inside
the binaries (import machinery needs them at runtime).
``tests/test_flatten_consistency.py`` pins digests ↔ manifest.

Constraints on group membership: tail names must be unique within a group
(one ``PyInit_<tail>`` each), and members must not need the
``embedsignature``/docstring carve-out (groups compile with the strip set).
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import sys
from pathlib import Path

_PKG_PREFIX = "depz_camera_sdk."
_SALT = b"depz-flatten-v1"

# fat extension file stem → salted digests of the member module names.
_MEMBER_DIGESTS: dict[str, frozenset[str]] = {
    "_camdrv": frozenset({
        "7dd3c2ae419cf356",
        "3eb185d3460d8e75",
        "0301f1d376db5abc",
        "c52629599c2e99a9",
        "4a0bbc94c169dfc0",
    }),
}

_EXT_SUFFIXES = (".so", ".pyd")


def _digest(fullname: str) -> str:
    return hashlib.blake2s(fullname.encode(), key=_SALT,
                           digest_size=8).hexdigest()


class _FlatModuleFinder:
    """sys.meta_path finder serving member modules from fat extensions."""

    def __init__(self, digest_to_so: dict[str, str]) -> None:
        self._map = digest_to_so

    def find_spec(self, fullname, path=None, target=None):  # noqa: D102
        if not fullname.startswith(_PKG_PREFIX):
            return None   # fast-exit: this finder sees EVERY import
        so = self._map.get(_digest(fullname))
        if so is None:
            return None
        loader = importlib.machinery.ExtensionFileLoader(fullname, so)
        return importlib.util.spec_from_file_location(fullname, so,
                                                      loader=loader)


def install() -> bool:
    """Install the finder for every group whose fat extension exists.
    Idempotent; returns True if at least one group is being served
    flattened. Called from ``depz_camera_sdk/__init__`` BEFORE any
    subpackage import, so member imports anywhere resolve through it."""
    if any(isinstance(f, _FlatModuleFinder) for f in sys.meta_path):
        return True
    pkg_root = Path(__file__).resolve().parent
    mapping: dict[str, str] = {}
    for stem, digests in _MEMBER_DIGESTS.items():
        fat_file = next(
            (p for p in sorted(pkg_root.glob(stem + ".*"))
             if p.suffix in _EXT_SUFFIXES),
            None,
        )
        if fat_file is None:
            continue  # pure-python checkout — normal imports handle members
        for d in digests:
            mapping[d] = str(fat_file)
    if not mapping:
        return False
    sys.meta_path.insert(0, _FlatModuleFinder(mapping))
    return True
