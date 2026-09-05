"""Shared logging utilities — throttled exception logging + milestone steps.

Plain-.py module (NOT in ``_compile_manifest.COMPILE``) so the viewer can
import it too and so hot-reload / tests never fight a compiled shadow.

Three tools, one policy — **no silent failures, no log floods**:

- :class:`ThrottledLogger` — for per-frame / hot-loop failure paths. First
  occurrence of a key logs immediately WITH traceback; repeats within the
  window are counted; window expiry emits one summary (with the latest
  traceback) carrying the suppressed count. A camera that starts failing
  every frame at 30 fps produces ~1 log line per 10 s, not 300.
- :func:`log_once` — for one-shot degradation notices ("controls
  unavailable: camera sliders disabled"). Logs the first time per key,
  never again for the process lifetime.
- :func:`log_step` — success milestones in one greppable convention::

      grep "STEP ok:" logs/<session>/*.log

  shows the full startup ladder (hardware-scan → runtimes → camera-info →
  camera-open → model-load → pipeline-start → first-frame → ...).
"""

import logging
import sys
import threading
import time


class ThrottledLogger:
    """Per-key rate-limited exception logging for hot paths.

    Thread-safe; monotonic-clock based. Call :meth:`log_exception` from
    inside an ``except`` block — the active exception's traceback is
    attached to the first occurrence and to each window summary.
    """

    def __init__(self, interval_s: float = 10.0) -> None:
        self._interval = float(interval_s)
        self._lock = threading.Lock()
        # key -> [window_start_monotonic, suppressed_count]
        self._state: dict = {}

    def log_exception(self, logger: logging.Logger, key: str,
                      msg: str, *args, level: int = logging.ERROR) -> None:
        """Log ``msg % args`` at ``level`` with the current traceback —
        subject to per-``key`` throttling (see class docstring)."""
        now = time.monotonic()
        with self._lock:
            st = self._state.get(key)
            if st is None:
                self._state[key] = [now, 0]
                suffix = ""
            elif now - st[0] >= self._interval:
                elapsed = now - st[0]
                suppressed = st[1]
                self._state[key] = [now, 0]
                suffix = (f" (repeated {suppressed} more times in last "
                          f"{elapsed:.1f}s)") if suppressed else ""
            else:
                st[1] += 1
                return
        # exc_info only if we're actually inside an except block —
        # avoids the confusing "NoneType: None" trailer otherwise.
        exc_info = sys.exc_info()[0] is not None
        logger.log(level, msg + suffix, *args, exc_info=exc_info)


# Process-wide default throttle. Call sites share it with namespaced keys
# ("camera_source.grab", "ws.stats.recording_status", ...) so importing
# modules don't each need to manage an instance.
THROTTLE = ThrottledLogger()


_ONCE_LOCK = threading.Lock()
_ONCE_KEYS: set = set()


def log_once(logger: logging.Logger, key: str, msg: str, *args,
             level: int = logging.WARNING, exc_info: bool = False) -> None:
    """Log ``msg % args`` the first time ``key`` is seen in this process;
    subsequent calls are no-ops. For one-shot degradation notices where
    repeating adds noise but silence would hide real behavior change."""
    with _ONCE_LOCK:
        if key in _ONCE_KEYS:
            return
        _ONCE_KEYS.add(key)
    if exc_info and sys.exc_info()[0] is None:
        exc_info = False
    logger.log(level, msg, *args, exc_info=exc_info)


def reset_once_keys() -> None:
    """Forget all :func:`log_once` keys. Test helper."""
    with _ONCE_LOCK:
        _ONCE_KEYS.clear()


def log_step(logger: logging.Logger, area: str, detail: str = "",
             **fields) -> None:
    """Emit a success-milestone line: ``STEP ok: <area>: <detail> k=v ...``.

    One consistent, greppable convention for "this stage worked" across
    the SDK and viewer. ``fields`` are appended as ``key=value`` pairs.
    """
    parts = []
    if detail:
        parts.append(str(detail))
    parts.extend(f"{k}={v}" for k, v in fields.items())
    tail = " ".join(parts)
    if tail:
        logger.info("STEP ok: %s: %s", area, tail)
    else:
        logger.info("STEP ok: %s", area)
