"""Tell an application what an engine is busy with while it starts.

Most of an engine's start-up is quick. Two steps can take long and an
application showing a status wants to name them:

* ``"preparing"`` — the engine is being created for a model (any engine:
  ONNX Runtime sessions, OpenVINO compiles, TensorRT loading a plan). On a
  first run this can take a minute or more; later runs are quick.
* ``"building"`` — TensorRT has no plan for this model, GPU and size yet
  and is building one: minutes, once.

An application registers one listener with :func:`set_listener`; it is
called as ``listener(event, info)`` where ``event`` is ``"start"`` or
``"end"`` and ``info`` holds ``{"step": "preparing" | "building",
"engine": <slug>}``. Steps nest — ``"building"`` runs inside
``"preparing"``. A listener that raises is ignored: reporting must never
break a start.
"""
from __future__ import annotations

import contextlib
import logging
from typing import Callable, Iterator, Optional

_LOGGER = logging.getLogger(__name__)

Listener = Callable[[str, dict], None]
_listener: Optional[Listener] = None


def set_listener(listener: Optional[Listener]) -> None:
    """Register the function told about long engine steps (None removes it)."""
    global _listener
    _listener = listener


def _tell(event: str, info: dict) -> None:
    fn = _listener
    if fn is None:
        return
    try:
        fn(event, dict(info))
    except Exception:
        _LOGGER.debug("engine progress listener raised", exc_info=True)


@contextlib.contextmanager
def step(name: str, engine: str) -> Iterator[None]:
    """Report ``name`` as started on entry and ended on exit (even on error)."""
    info = {"step": name, "engine": engine}
    _tell("start", info)
    try:
        yield
    finally:
        _tell("end", info)
