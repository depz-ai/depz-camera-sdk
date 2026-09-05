"""SDK-wide logging bootstrap — console + per-session rotating log files.

Plain-.py module (NOT in ``_compile_manifest.COMPILE``). Nothing here runs
at import time — the SDK stays a well-behaved library. Entry points opt in:

- the viewer server calls :func:`configure_logging` in ``__main__.main()``,
- the pipeline-host child calls it first thing in ``run_pipeline_host``
  (reusing the parent's session via :func:`get_child_log_config`),
- SDK-standalone users call :func:`enable_default_logging`.

Log directory resolution (:func:`resolve_log_dir`), first match wins:

  1. explicit argument (``--log-dir`` CLI flag)
  2. ``$DEPZ_LOG_DIR`` environment variable
  3. ``<cwd>/logs`` when ``<cwd>/.git`` exists — running from a repo
     checkout; the repo's ``.gitignore`` already ignores ``logs``
  4. per-OS user log dir (installed wheels / one-click installers, where
     cwd is the app install dir and — on Linux .desktop launches — no
     console is visible at all):

       Windows  %LOCALAPPDATA%\\depz\\Logs
       macOS    ~/Library/Logs/depz
       Linux    $XDG_STATE_HOME/depz/logs  (or ~/.local/state/depz/logs)

Session layout — one directory per app run, one file per process (a shared
FileHandler across processes is unsafe on Windows):

    <log_root>/<session>/
        server.log               viewer server process
        pipeline-host-<pid>.log  each pipeline child (pid-suffixed)
        ui.log                   forwarded browser errors
        sdk.log                  standalone-SDK runs

Console keeps the pre-existing format; files add process/thread names and
always capture DEBUG regardless of the console level. Old sessions are
pruned (newest :data:`KEEP_SESSIONS` kept). ``sys.excepthook`` /
``threading.excepthook`` are installed so no uncaught exception can die
invisibly, and ``logging.captureWarnings`` routes warnings into the logs.
"""

import logging
import logging.handlers
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Union


CONSOLE_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
FILE_FORMAT = ("%(asctime)s %(levelname)s [%(processName)s %(threadName)s] "
               "%(name)s: %(message)s")

#: Newest N session directories kept by the best-effort prune.
KEEP_SESSIONS = 15

_FILE_MAX_BYTES = 20_000_000  # per-file cap; runaway loops can't fill disk
_FILE_BACKUPS = 2

# Active configuration — set once by configure_logging (first call wins),
# read by get_child_log_config so the pipeline-host child can join the
# parent's session directory.
_ACTIVE: Optional[dict] = None
_LOCK = threading.Lock()

_LOGGER = logging.getLogger("depz_camera_sdk.logging_setup")


def _user_log_dir() -> Path:
    """Per-platform user log directory, mirroring the cache-dir convention
    the viewer already uses (``%LOCALAPPDATA%\\depz\\Cache`` etc.)."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / "depz"
    if os.name == "nt":
        local_app_data = Path(os.environ.get("LOCALAPPDATA")
                              or Path.home() / "AppData" / "Local")
        return local_app_data / "depz" / "Logs"
    state_home = os.environ.get("XDG_STATE_HOME")
    base = Path(state_home) if state_home else Path.home() / ".local" / "state"
    return base / "depz" / "logs"


def resolve_log_dir(explicit: Union[str, Path, None] = None) -> Path:
    """Resolve the log ROOT (sessions are subdirectories of this). See
    module docstring for the precedence chain."""
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get("DEPZ_LOG_DIR")
    if env:
        return Path(env).expanduser()
    cwd = Path.cwd()
    if (cwd / ".git").exists():
        return cwd / "logs"
    return _user_log_dir()


def configure_logging(*, component: str,
                      level: Union[str, int] = "INFO",
                      log_dir: Union[str, Path, None] = None,
                      session: Optional[str] = None,
                      console: bool = True) -> Path:
    """Configure root logging for this process. Idempotent — the first
    call wins; later calls return the active session dir unchanged.

    ``component`` names this process's log file (``"server"``,
    ``"pipeline-host"`` — pid-suffixed, ``"sdk"``). ``level`` sets the
    CONSOLE threshold; the file handler always records DEBUG. ``session``
    lets a child process join its parent's session directory (see
    :func:`get_child_log_config`).

    Returns the session directory in use. If the log directory cannot be
    created (read-only install, exotic sandbox), logging degrades to
    console-only rather than failing startup — that failure itself is
    logged.
    """
    global _ACTIVE
    with _LOCK:
        if _ACTIVE is not None:
            return Path(_ACTIVE["session_dir"])

        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)

        root = logging.getLogger()
        # File handler records everything; console filters at `level`.
        root.setLevel(logging.DEBUG)

        if console:
            ch = logging.StreamHandler(sys.stderr)
            ch.setLevel(level)
            ch.setFormatter(logging.Formatter(CONSOLE_FORMAT))
            root.addHandler(ch)

        log_root = resolve_log_dir(log_dir)
        own_session = session is None
        session = session or (time.strftime("%Y%m%d-%H%M%S")
                              + f"-{os.getpid()}")
        session_dir = log_root / session

        filename = component
        if component == "pipeline-host":
            # Several children can run within one session — pid-suffix so
            # they never share a file (unsafe on Windows).
            filename = f"{component}-{os.getpid()}"

        try:
            session_dir.mkdir(parents=True, exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(
                session_dir / f"{filename}.log",
                maxBytes=_FILE_MAX_BYTES, backupCount=_FILE_BACKUPS,
                encoding="utf-8", delay=True)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(FILE_FORMAT))
            root.addHandler(fh)
        except OSError:
            _LOGGER.exception(
                "cannot create log dir %s — continuing console-only",
                session_dir)

        logging.captureWarnings(True)
        _install_excepthooks()

        # The file handler records DEBUG for depz code — but a few
        # third-party libraries are pathologically chatty at DEBUG
        # (comtypes traces every COM AddRef/Release, hundreds of lines
        # per device scan). Cap them at INFO so the session logs stay
        # readable; real problems from these libs surface at WARNING+.
        for noisy in ("comtypes", "urllib3", "PIL", "matplotlib",
                      "asyncio", "charset_normalizer"):
            logging.getLogger(noisy).setLevel(logging.INFO)

        _ACTIVE = {
            "component": component,
            "level": logging.getLevelName(level),
            "log_root": str(log_root),
            "session": session,
            "session_dir": str(session_dir),
        }

        # Only the session OWNER prunes (children joining a session must
        # never race the parent's prune against their own mkdir).
        if own_session:
            _prune_sessions(log_root, keep=KEEP_SESSIONS,
                            current=session_dir)

    _LOGGER.info("logging to %s (component=%s, console=%s)",
                 session_dir, component, logging.getLevelName(level))
    return session_dir


def get_child_log_config() -> Optional[dict]:
    """Active config for handing to a child process:
    ``{"log_dir", "session", "level"}`` — or ``None`` when
    :func:`configure_logging` was never called. The child passes these to
    its own :func:`configure_logging` so its file lands in the same
    session directory."""
    if _ACTIVE is None:
        return None
    return {
        "log_dir": _ACTIVE["log_root"],
        "session": _ACTIVE["session"],
        "level": _ACTIVE["level"],
    }


def active_session_dir() -> Optional[Path]:
    """Session directory of the active configuration, or ``None``."""
    if _ACTIVE is None:
        return None
    return Path(_ACTIVE["session_dir"])


def attach_logger_file(logger_name: str, filename: str) -> Optional[Path]:
    """Give one named logger its own rotating file inside the active
    session directory (e.g. forwarded browser errors →
    ``depz_camera_viewer.client`` → ``ui.log``). Propagation stays on, so
    records still reach the console and the component log. Returns the
    file path, or ``None`` when logging is unconfigured / console-only."""
    if _ACTIVE is None:
        return None
    session_dir = Path(_ACTIVE["session_dir"])
    path = session_dir / filename
    try:
        session_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            path, maxBytes=_FILE_MAX_BYTES, backupCount=_FILE_BACKUPS,
            encoding="utf-8", delay=True)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(FILE_FORMAT))
        logging.getLogger(logger_name).addHandler(fh)
        return path
    except OSError:
        _LOGGER.exception("cannot attach %s to logger %s",
                          filename, logger_name)
        return None


def enable_default_logging(level: Union[str, int] = "INFO") -> Path:
    """Opt-in convenience for SDK-standalone users::

        import depz_camera_sdk
        depz_camera_sdk.enable_default_logging()

    Console + ``<log_root>/<session>/sdk.log``. Returns the session dir.
    Nothing is configured at import time — this call is the opt-in."""
    return configure_logging(component="sdk", level=level)


def reset_logging_for_tests() -> None:
    """Drop the active configuration and detach handlers installed by
    :func:`configure_logging`. Test helper — production code never needs
    to unconfigure."""
    global _ACTIVE
    with _LOCK:
        _ACTIVE = None
        root = logging.getLogger()
        for h in list(root.handlers):
            root.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass


# ── internals ────────────────────────────────────────────────────────


def _prune_sessions(log_root: Path, *, keep: int, current: Path) -> None:
    """Best-effort: keep the newest ``keep`` session dirs (by name — the
    timestamp prefix sorts chronologically), never touching ``current``.
    Failures are logged at DEBUG and ignored — pruning must never take
    down startup."""
    try:
        sessions = sorted(
            (d for d in log_root.iterdir() if d.is_dir()),
            key=lambda d: d.name, reverse=True)
    except OSError:
        return
    for stale in sessions[keep:]:
        if stale == current:
            continue
        try:
            for f in stale.iterdir():
                f.unlink()
            stale.rmdir()
        except OSError:
            _LOGGER.debug("could not prune old log session %s",
                          stale, exc_info=True)


def _install_excepthooks() -> None:
    """Route uncaught exceptions (main thread + worker threads) through
    logging so they reach the log files, not just a possibly-invisible
    stderr. KeyboardInterrupt keeps the default quiet exit."""

    def _sys_hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        logging.getLogger("uncaught").critical(
            "uncaught exception", exc_info=(exc_type, exc, tb))

    def _thread_hook(args):
        if issubclass(args.exc_type, SystemExit):
            return
        logging.getLogger("uncaught").critical(
            "uncaught exception in thread %r",
            args.thread.name if args.thread else "?",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    sys.excepthook = _sys_hook
    threading.excepthook = _thread_hook
