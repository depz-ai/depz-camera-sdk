"""HTTP download + cross-process locking primitives for the model cache.

Kept apart from :mod:`.base` so the catalog types there stay about
*what* a variant needs, and this module about *how* bytes get onto disk
safely: streamed with a socket timeout, counted against what the server
promised, and serialized across processes that want the same file.
"""

import hashlib
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

# Some catalog hosts (depz.ai behind a WAF) reject the default
# ``Python-urllib/x.y`` agent with 403 — send an explicit one.
HTTP_USER_AGENT = "depz-camera-sdk"

# Socket timeout: the longest a connect or a single read may stall. Not
# a cap on the whole transfer — a large model on a slow link is fine as
# long as bytes keep arriving.
DEFAULT_TIMEOUT_S = 30.0

_CHUNK_BYTES = 1 << 20


class DownloadError(RuntimeError):
    """A download failed. ``status`` is the HTTP status code when the
    server answered with an error, else ``None`` (network failure,
    timeout, truncated body)."""

    def __init__(self, message: str, *, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status


def http_download(url: str, dest: Path, *,
                  progress_cb: "Optional[Callable[[int, int], None]]" = None,
                  expected_size: Optional[int] = None,
                  timeout: float = DEFAULT_TIMEOUT_S) -> tuple[int, str]:
    """Stream ``url`` into ``dest`` and return ``(bytes_written, sha256)``.

    ``progress_cb(done, total)`` is called after every chunk; ``total``
    is the server's Content-Length, falling back to ``expected_size``,
    else 0 (unknown). Raises :class:`DownloadError` on an HTTP error
    status, a network failure, or a body shorter/longer than the
    Content-Length the server announced."""
    req = urllib.request.Request(url, headers={"User-Agent": HTTP_USER_AGENT})
    digest = hashlib.sha256()
    done = 0
    announced = 0
    callback_error: Optional[BaseException] = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, \
                open(dest, "wb") as out:
            try:
                announced = int(resp.headers.get("Content-Length") or 0)
            except (TypeError, ValueError):
                announced = 0
            total = announced or int(expected_size or 0)
            while True:
                chunk = resp.read(_CHUNK_BYTES)
                if not chunk:
                    break
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if progress_cb is not None:
                    try:
                        progress_cb(done, total)
                    except BaseException as e:  # noqa: BLE001
                        # The caller's own exception (e.g. a cancel) —
                        # must reach it unchanged, not as a DownloadError.
                        callback_error = e
                        break
    except urllib.error.HTTPError as e:
        raise DownloadError(
            f"HTTP {e.code} {e.reason} for {url}", status=e.code) from e
    except Exception as e:
        # DNS / refused / TLS (URLError), socket timeout or reset
        # (OSError), body cut short mid-chunk (http.client.IncompleteRead),
        # disk full while writing.
        raise DownloadError(f"{type(e).__name__}: {e} for {url}") from e
    if callback_error is not None:
        raise callback_error
    if announced and done != announced:
        raise DownloadError(
            f"incomplete download: got {done} of {announced} bytes "
            f"from {url}")
    return done, digest.hexdigest()


class FileLock:
    """Advisory cross-process mutex backed by an OS lock on ``path``.

    The operating system releases the lock when the holder exits — even
    on a crash or ``kill -9`` — so there are no stale locks to detect.
    The lock file itself is left in place on purpose: deleting it while
    another process waits on it would let two holders in.

    Use as a context manager. ``timeout=None`` waits indefinitely;
    otherwise :class:`TimeoutError` is raised after ``timeout`` seconds.
    """

    def __init__(self, path: Path, *, timeout: Optional[float] = None,
                 poll_interval: float = 0.1) -> None:
        self._path = Path(path)
        self._timeout = timeout
        self._poll = poll_interval
        self._fh = None

    def __enter__(self) -> "FileLock":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self._path, "a+b")
        deadline = (None if self._timeout is None
                    else time.monotonic() + self._timeout)
        try:
            while not _try_lock(fh):
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"could not lock {self._path} within "
                        f"{self._timeout:.1f}s — another process holds it")
                time.sleep(self._poll)
        except BaseException:
            fh.close()
            raise
        self._fh = fh
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        fh, self._fh = self._fh, None
        if fh is None:
            return
        try:
            _unlock(fh)
        finally:
            fh.close()


if sys.platform == "win32":
    import msvcrt

    def _try_lock(fh) -> bool:
        # Byte-range lock on the first byte (locking past EOF is legal on
        # Windows, so the file may stay empty).
        fh.seek(0)
        try:
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    def _unlock(fh) -> None:
        fh.seek(0)
        try:
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
else:
    import fcntl

    def _try_lock(fh) -> bool:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        return True

    def _unlock(fh) -> None:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass


def unique_temp_path(target: Path) -> Path:
    """A fresh, already-created temp file next to ``target`` (same
    filesystem, so the final ``replace`` is atomic). Named
    ``<target>.<random>.part``."""
    import tempfile
    fd, name = tempfile.mkstemp(prefix=target.name + ".", suffix=".part",
                                dir=target.parent)
    os.close(fd)
    return Path(name)


def remove_stale_parts(target: Path) -> None:
    """Delete ``<target>.*.part`` leftovers. Only call while holding the
    target's :class:`FileLock` — then no live download owns them."""
    stale = list(target.parent.glob(target.name + ".*.part"))
    stale.append(target.with_name(target.name + ".part"))
    for p in stale:
        try:
            p.unlink()
        except OSError:
            pass
