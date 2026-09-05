"""InferenceEngine + InferenceFuture ABCs.

Common contract for OpenVINO and ONNX backends. Both engines submit
work asynchronously and return a Future-like handle that the pipeline
worker polls between iterations. Designed to map 1:1 to std::future
in a future C++ port.
"""

import threading
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class AbortedInference(RuntimeError):
    """The in-flight inference was aborted via :meth:`InferenceEngine.abort`
    (user stop/cancel). NOT an engine fault — pipeline workers treat it as
    a clean exit, never as a crash to report."""


class InferenceFuture(ABC):
    """Result handle for one in-flight inference."""

    @abstractmethod
    def done(self) -> bool: ...

    @abstractmethod
    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block up to `timeout` seconds. Return True if completed."""

    @abstractmethod
    def result(self) -> np.ndarray:
        """Return raw disparity output (h_canvas × w_canvas float32).
        Raises if errored or not yet done."""


class InferenceEngine(ABC):
    """Stereo inference backend.

    Lifecycle:
        engine = SomeEngine(model_path, ...)
        engine.warmup()
        future = engine.submit(*blobs)
        ...
        if future.done(): disp = future.result()
        ...
        engine.shutdown()

    Most stereo nets have 2 image inputs (L, R). Multi-stage variants
    (e.g. CREStereo "combined" with init/next pairs at two resolutions)
    declare 4 inputs. The architecture's `preprocess` produces blobs in
    the order the model declares them; the engine just forwards them.

    Thread-safety:
        - submit() / shutdown() called from the pipeline worker thread only.
        - InferenceFuture.done()/result() may be called from the same
          worker — no other thread should touch them.
    """

    # Set by :func:`make_engine` after construction (the registry knows
    # the canonical slug; the engine subclass only knows its device/EP).
    # Empty string when constructed directly (test mocks, custom code
    # bypassing the registry).
    slug: str = ""

    def __init__(self, model_path: str) -> None:
        # Public attribute — `Model.model_path` reads it through.
        # Subclasses must call ``super().__init__(model_path)``.
        self.model_path: str = model_path

    @property
    @abstractmethod
    def input_shapes(self) -> tuple[tuple[int, ...], ...]:
        """Per-input NCHW shapes in declared model order. The single
        engine-introspection primitive: warmup uses it to allocate
        zero-filled blobs, and :attr:`input_size` derives the canvas
        H×W from it."""

    @property
    def input_size(self) -> tuple[int, int]:
        """(canvas_h, canvas_w) of the *largest* model input — the
        architecture's canvas size. Multi-resolution models report
        the full-res input; downsampled inputs are produced by the
        architecture's preprocess from the same canvas."""
        h, w = max((s[-2], s[-1]) for s in self.input_shapes)
        return h, w

    @abstractmethod
    def submit(self, *blobs: np.ndarray) -> InferenceFuture:
        """Submit one inference. The number/shape of blobs must match
        :attr:`input_shapes` in declared order."""

    def abort(self) -> None:
        """Abort the in-flight inference NOW (milliseconds, not minutes).
        Terminal for the engine instance — only the stop/cancel path
        calls it, right before :meth:`shutdown`. The aborted future
        fails with :class:`AbortedInference`. Default no-op (classical
        backends run synchronously and have nothing in flight)."""

    @abstractmethod
    def shutdown(self, abandon: bool = False) -> None:
        """Release resources. ``abandon=True`` (the stop path, after
        :meth:`abort`) must NOT block on in-flight work — the process
        is about to exit and any straggler native thread dies with it."""

    def warmup(self) -> None:
        """Pre-compile + first inference. Submits one all-zeros tensor
        per declared input. GPU/NPU first-compile can take 30–60 s;
        OpenVINO caches the result on disk so subsequent loads are
        sub-second. Override only if a backend has a cheaper warmup."""
        zeros = tuple(np.zeros(s, dtype=np.float32) for s in self.input_shapes)
        fut = self.submit(*zeros)
        fut.wait()
        fut.result()


# ── Common simple Future built around a threading.Event ──

class EventFuture(InferenceFuture):
    """Generic Future for engines that fire a callback on completion
    (OpenVINO AsyncInferQueue, ThreadPoolExecutor.submit's wrapper).

    Two-sided contract:
    - **Consumer side** (pipeline) sees only the
      :class:`InferenceFuture` interface (`done` / `wait` / `result`).
    - **Engine side** (subclass / engine implementer) calls exactly one
      of :meth:`set_result` / :meth:`set_error` when inference
      completes. Both are idempotent in the sense that the second call
      is a no-op (once the event is set the future is done).
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._result: Optional[np.ndarray] = None
        self._error: Optional[BaseException] = None

    def done(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._event.wait(timeout=timeout)

    def result(self) -> np.ndarray:
        if not self._event.is_set():
            raise RuntimeError("future not yet complete")
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result

    # ── Engine-side completion API ──────────────────────────────

    def set_result(self, result: np.ndarray) -> None:
        """Mark the future complete with a result. Engine-side."""
        self._result = result
        self._event.set()

    def set_error(self, exc: BaseException) -> None:
        """Mark the future failed. Engine-side."""
        self._error = exc
        self._event.set()
