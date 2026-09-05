"""All inference base classes — public ABCs + intermediate bases.

This module hosts every reusable base. Concrete inferences live in
``inferences/<network>.py``, one file per network.

Hierarchy::

    DepthInferenceBase                 ← public ABC, pipeline-facing,
    │                                    catalog metadata (LABEL,
    │                                    NEURAL, variants(), …)
    ├── NeuralInferenceBase            ← engine + abstract pre/post
    │   ├── NeuralStereoInferenceBase  ← stereo intermediate (OUTPUTS_DEPTH=False)
    │   └── NeuralMonoInferenceBase    ← mono intermediate (OUTPUTS_DEPTH=True,
    │                                     plus depth_scale/offset controls)
    ├── ClassicStereoInferenceBase     ← cv2-matcher scaffold
    └── (RectifyOnlyInference, lives in inferences/rectify_only.py)

The descriptor (:class:`Control`) and the get/set/coerce
machinery are shared with :class:`~depz_camera_sdk.filters.base.Filter`
via :class:`~depz_camera_sdk.._controls.ControlOwner`."""

import hashlib
import json
import logging
import re
import shutil
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Optional

import cv2
import numpy as np

from .._controls import FloatControl, Control, ControlOwner
from ..calibration import CanvasLayout
from ..engines import EventFuture, InferenceEngine, InferenceFuture

__all__ = [
    "Control",
    "DepthInferenceBase",
    "NeuralStereoInferenceBase",
    "NeuralMonoInferenceBase",
    "InferenceVariant",
    "ModelFile",
]


_LOGGER = logging.getLogger("depz_camera_sdk.inference")


# ───────────────────────── Slug → URL resolver ────────────────────────
# The catalog endpoint + per-process slug cache. ``ModelFile._resolve_url``
# reads/writes ``_slug_cache`` so a slug only round-trips once per
# session. Tests patch :data:`_SLUG_RESOLVER_URL` (or seed
# ``_slug_cache`` directly) to bypass the network.


_SLUG_RESOLVER_URL = "https://depz.ai/api/public/get-model-file-link"

# Some catalog hosts (depz.ai behind a WAF) reject the default
# ``Python-urllib/x.y`` agent with 403 — send an explicit one.
_HTTP_USER_AGENT = "depz-camera-sdk"


# Google Drive share links (``…/file/d/<id>/view`` or ``…?id=<id>``) can't
# be fetched by urllib directly — they return an HTML page, not the file.
# Rewrite them to the usercontent direct-download form, which streams the
# bytes (``confirm=t`` also bypasses the >100 MB virus-scan interstitial).
_GDRIVE_FILE_RE = re.compile(r"drive\.google\.com/file/d/([^/?#]+)")
_GDRIVE_ID_RE = re.compile(r"[?&]id=([^&]+)")


def _direct_download_url(url: str) -> str:
    """Return a URL urllib can fetch directly. Rewrites Google Drive
    share links to the usercontent download endpoint; passes every other
    URL through unchanged."""
    if "drive.google.com" not in url:
        return url
    m = _GDRIVE_FILE_RE.search(url) or _GDRIVE_ID_RE.search(url)
    if not m:
        return url
    file_id = m.group(1)
    return ("https://drive.usercontent.google.com/download"
            f"?id={file_id}&export=download&confirm=t")


_slug_cache: dict[str, str] = {}


# Model grade/tag metadata lives inline next to each variant
# (``configurations["selector_meta"]`` in ``inference/inferences/*.py``) and
# is read directly by the viewer's model selector — there is no metadata
# API. (Model *files* resolve via ``get-model-file-link`` above; grades do
# not.)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _default_stderr_progress() -> Callable[[int, int], None]:
    """One-line stderr progress bar — written when no explicit
    ``download_progress_cb`` is supplied. Throttled to ~10 Hz so it doesn't
    flood logs; clears its line on completion."""
    state = {"last": 0.0}

    def _cb(done: int, total: int) -> None:
        now = time.monotonic()
        if total and done < total and (now - state["last"]) < 0.1:
            return
        state["last"] = now
        if total:
            pct = 100.0 * done / total
            mb_d, mb_t = done / (1 << 20), total / (1 << 20)
            line = f"  {pct:5.1f}%  {mb_d:7.1f} / {mb_t:7.1f} MB"
        else:
            line = f"  {done / (1 << 20):7.1f} MB"
        end = "\n" if (total and done >= total) else ""
        print(f"\r{line}", end=end, file=sys.stderr, flush=True)

    return _cb


# ──────────────────────── ModelFile + InferenceVariant ────────────────


@dataclass(frozen=True, slots=True)
class ModelFile:
    """One file an :class:`InferenceVariant` needs at runtime.

    ``slug`` is the model-registry identifier; the SDK resolves it
    to a public download URL at fetch time via the catalog service.
    Slugs are stable — moving a model between mirrors / CDNs / buckets
    is a server-side edit, not a client-side code change.

    ``filename`` is the on-disk basename inside the variant cache
    directory. For archive downloads (``archive_member`` set), this
    is the *extracted* filename — the archive itself is discarded.

    ``sha256`` is optional but strongly recommended — it's the only
    way :meth:`InferenceVariant.resolve_files` can tell a corrupted-
    but-right-size file from a good one. ``size_bytes`` is advisory:
    surfaced to the UI for download-progress totals before the HTTP
    response is read.

    ``archive_member`` extracts a single file from an archive payload
    (zip currently supported). The resolved URL points at an archive,
    we open it in-memory / via tempfile, copy the named member into
    ``filename``, and discard the archive — variant cache stays small.
    """

    slug: str
    filename: str
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None
    archive_member: Optional[str] = None

    def is_cached(self, target: Path) -> bool:
        """True iff ``target`` exists with matching size + sha256
        (when declared). Read-only — never opens a socket. On the hot
        read path we don't recompute sha256: it's verified at download
        time, and a corrupted-but-right-size file is rare enough that
        it surfaces as an ONNX-load error one layer up."""
        if not target.is_file():
            return False
        st = target.stat()
        if self.size_bytes is not None and st.st_size != self.size_bytes:
            return False
        return True

    def resolve(
            self,
            target: Path,
            download_progress_cb: "Optional[Callable[[int, int], None]]" = None,
    ) -> None:
        """Ensure the file is on disk at ``target``: no-op if
        :meth:`is_cached`, otherwise :meth:`download`."""
        if self.is_cached(target):
            return
        self.download(target, download_progress_cb)

    def _resolve_url(self, *, timeout: float = 10.0) -> str:
        """Resolve :attr:`slug` to its public download URL via the SDK
        catalog service. Cached per process in :data:`_slug_cache`.

        An explicit User-Agent is sent because depz.ai (behind a WAF)
        rejects the default ``Python-urllib`` agent with 403."""
        cached = _slug_cache.get(self.slug)
        if cached is not None:
            return cached
        qs = urllib.parse.quote(self.slug, safe="")
        url = f"{_SLUG_RESOLVER_URL}?slug={qs}"
        req = urllib.request.Request(url, headers={"User-Agent": _HTTP_USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            raise RuntimeError(
                f"failed to resolve slug {self.slug!r}: {e}") from e
        link = body.get("link") if isinstance(body, dict) else None
        if not isinstance(link, str) or not link:
            raise RuntimeError(
                f"failed to resolve slug {self.slug!r}: no link in {body!r}")
        _slug_cache[self.slug] = link
        return link

    def download(
            self,
            target: Path,
            download_progress_cb: "Optional[Callable[[int, int], None]]" = None,
    ) -> None:
        """Resolve :attr:`slug` to a URL, fetch it, and land the result
        at ``target`` atomically. Direct file download by default;
        archive-extract flow when :attr:`archive_member` is set."""
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".part")
        if tmp.exists():
            tmp.unlink()

        url = _direct_download_url(self._resolve_url())
        _LOGGER.info("downloading %s (slug=%s) → %s", url, self.slug, target)

        def _make_hook():
            if download_progress_cb is None:
                return None
            def _hook(blocks: int, block_size: int, total_size: int) -> None:
                done = blocks * block_size
                if total_size > 0:
                    done = min(done, total_size)
                download_progress_cb(done, total_size if total_size > 0 else (
                    self.size_bytes or 0))
            return _hook

        try:
            if self.archive_member is None:
                hook = _make_hook()
                if hook is not None:
                    urllib.request.urlretrieve(url, str(tmp), hook)
                else:
                    urllib.request.urlretrieve(url, str(tmp))
            else:
                with tempfile.TemporaryDirectory(
                        prefix="depz-archive-", dir=target.parent) as td:
                    arch_path = Path(td) / "archive"
                    hook = _make_hook()
                    if hook is not None:
                        urllib.request.urlretrieve(url, str(arch_path), hook)
                    else:
                        urllib.request.urlretrieve(url, str(arch_path))
                    with zipfile.ZipFile(arch_path) as zf:
                        member = self.archive_member
                        try:
                            info = zf.getinfo(member)
                        except KeyError:
                            info = next(
                                (i for i in zf.infolist()
                                 if Path(i.filename).name == member),
                                None)
                            if info is None:
                                raise RuntimeError(
                                    f"archive {url} has no member "
                                    f"{member!r}; available: "
                                    f"{[i.filename for i in zf.infolist() if not i.is_dir()]}")
                        with zf.open(info) as src, tmp.open("wb") as dst:
                            shutil.copyfileobj(src, dst, length=1 << 20)

            if self.sha256 is not None:
                got = _sha256_file(tmp)
                if got != self.sha256:
                    raise RuntimeError(
                        f"sha256 mismatch for slug {self.slug!r}: "
                        f"expected {self.sha256}, got {got}")
            tmp.replace(target)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise


@dataclass(frozen=True, slots=True)
class InferenceVariant:
    """One concrete configuration of an inference.

    ``slug`` is the second identifier in the (inference, variant)
    pair — stable, URL-safe, unique within the inference's catalog.
    ``label`` is the human-facing string the UI shows in the dropdown.

    ``files_to_download`` lists every file the variant needs on disk.
    Empty tuple = nothing to fetch (classical algorithms — SGBM,
    StereoBM, rectify-only). For neural variants the **first** entry
    is the entry point the engine loads.

    ``inference_kwargs`` are forwarded verbatim to the inference's
    constructor by :func:`build_inference`
    (``cls(engine=..., **variant.inference_kwargs)``). Use this for
    per-variant ctor args — SGBM/StereoBM/RectifyOnly ``canvas_size``,
    etc.

    ``engine_kwargs`` are forwarded to :func:`make_engine` when the
    engine is built (``make_engine(slug, model_path, **engine_kwargs)``).
    Use this for per-variant engine controls (``input_shape`` for
    dynamic-shape ONNXes, ``providers``, ``cache_dir``).

    ``configurations`` is opaque to the framework — free-form metadata for the
    UI / debugging (``size``, ``dataset``, ``input_size``, …). Not
    passed to either the constructor or the engine.

    ``inference_slug`` is filled in by ``register_inference`` when the
    inference enumerates its variants — concrete subclasses don't pass
    it when constructing the dataclass."""

    slug: str
    label: str
    files_to_download: tuple[ModelFile, ...] = ()
    inference_kwargs: dict[str, Any] = field(default_factory=dict)
    engine_kwargs: dict[str, Any] = field(default_factory=dict)
    configurations: dict[str, Any] = field(default_factory=dict)
    _inference_slug: str = ""

    @property
    def inference_slug(self) -> str:
        """Inference identifier (e.g. ``"lightstereo"``). Raises
        ``ValueError`` when read on a variant whose registry binding
        hasn't happened (``_inference_slug`` is empty)."""
        if not self._inference_slug:
            raise ValueError(
                f"variant {self.slug!r} has no inference_slug bound — "
                "pass a variant returned by Inference.variants() / "
                "Inference.get_variant(), not one constructed in "
                "isolation")
        return self._inference_slug

    @property
    def cache_dir(self) -> Path:
        """Deterministic cache directory for this variant. Read-only
        path computation — does NOT create the directory. Layout:
        ``<cache_root>/<inference_slug>/<slug>/``."""
        from .registry import cache_root
        return cache_root() / self.inference_slug / self.slug

    @property
    def inference(self) -> "type":
        """The inference class this variant belongs to — looked up
        via :func:`get_inference` from :attr:`inference_slug`."""
        from .registry import get_inference
        return get_inference(self.inference_slug)

    @property
    def is_cached(self) -> bool:
        """True iff every file in :attr:`files_to_download` is on disk
        with matching size + sha256 (when declared)."""
        cache_dir = self.cache_dir
        return all(f.is_cached(cache_dir / f.filename)
                   for f in self.files_to_download)

    def resolve_files(
            self, *,
            download_progress_cb: "Optional[Callable[[int, int], None] | bool]" = None,
    ) -> list[Path]:
        """Ensure the variant's cache directory is populated, then
        return absolute paths of every file in
        :attr:`files_to_download` (entry-point first)."""
        cache_dir = self.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        if download_progress_cb is None:
            download_progress_cb = _default_stderr_progress()
        elif download_progress_cb is False:
            download_progress_cb = None
        for f in self.files_to_download:
            f.resolve(cache_dir / f.filename, download_progress_cb)
        return [cache_dir / f.filename for f in self.files_to_download]


def build_inference(
    inference_slug: str,
    inference_variant_slug: str,
    *,
    engine_slug: "Optional[str]" = None,
    extra_engine_kwargs: "dict | None" = None,
    extra_inference_kwargs: "dict | None" = None,
    download_progress_cb: "Optional[Callable[[int, int], None] | bool]" = None,
) -> "DepthInferenceBase":
    """One-shot: ``(inference_slug, variant_slug)`` → fully-initialized
    :class:`DepthInferenceBase`.

    1. :func:`get_inference` resolves the inference class, then
       :meth:`get_variant` looks up the variant.
    2. :meth:`InferenceVariant.resolve_files` populates the cache
       (downloads on first use; cache hit is read-only ``stat()``s).
    3. ``variant.inference_kwargs`` (merged with
       ``extra_inference_kwargs``) are forwarded to the constructor.
       ``variant.engine_kwargs`` (merged with ``extra_engine_kwargs``)
       are forwarded to :func:`make_engine`.
    4. Classical inferences (``cls.NEURAL = False``) are constructed
       directly. Neural ones get an :class:`InferenceEngine` built
       around the entry-point file and passed via ``engine=``.

    ``engine_slug`` defaults to :func:`recommend_engine` (best engine
    available on this host). ``download_progress_cb`` controls the
    cache-fill reporter — see :meth:`InferenceVariant.resolve_files`.
    """
    from .registry import get_inference
    cls = get_inference(inference_slug)
    variant = cls.get_variant(inference_variant_slug)
    files = variant.resolve_files(download_progress_cb=download_progress_cb)

    inference_kwargs = {**variant.inference_kwargs,
                        **(extra_inference_kwargs or {})}

    if not cls.NEURAL:
        return cls(**inference_kwargs)

    if not files:
        raise RuntimeError(
            f"variant {variant.slug!r} is neural ({cls.SLUG!r}) but "
            f"declares no files_to_download — no .onnx for the engine")
    if engine_slug is None:
        from ..engines import recommend_engine
        engine_slug = recommend_engine()
        if engine_slug is None:
            raise RuntimeError("no inference engines available on this host")

    # Inferences whose ONNX is dynamic-shape declare INPUT_SIZE so the
    # engine reshapes to a concrete (1, 3, H, W) before compile.
    # Multi-head exports (Metric3D V2 → predicted_depth + normals +
    # confidence) declare OUTPUT_NAME so the engine binds to the right
    # output instead of the default first-output index.
    from ..engines import make_engine
    engine_kwargs = {**variant.engine_kwargs, **(extra_engine_kwargs or {})}
    if cls.INPUT_SIZE is not None and "input_shape" not in engine_kwargs:
        h, w = cls.INPUT_SIZE
        engine_kwargs["input_shape"] = (1, 3, h, w)
    if cls.OUTPUT_NAME is not None and "output_name" not in engine_kwargs:
        engine_kwargs["output_name"] = cls.OUTPUT_NAME
    engine = make_engine(engine_slug, str(files[0]), **engine_kwargs)

    return cls(engine=engine, **inference_kwargs)


class DepthInferenceBase(ABC, ControlOwner):
    """Stereo disparity computation — neural, classical, or no-op.

    Lifecycle (driven by :class:`DepthPipeline`):
      - construction: cheap. No model load, no matcher allocation.
      - :meth:`warmup`: heavy compile / allocation. Runs once on
        ``pipeline.start()``.
      - per-frame: pipeline calls :meth:`submit` with the rectified
        canvas pair, polls the returned :class:`InferenceFuture`, then
        calls :meth:`postprocess` on the raw output.
      - :meth:`shutdown`: release native resources. Idempotent.

    Controls:
      - :meth:`controls` returns the static descriptor list — drives UI
        introspection and the default :meth:`set` clamp logic (inherited
        from :class:`~depz_camera_sdk.._controls.ControlOwner`).
      - :meth:`set(name, value)` mutates one control. Default impl
        coerces + clamps via the descriptor and stores via setattr.
        Subclasses override to add side effects (e.g. SGBM rebuilds
        its matcher when controls change) — call ``super().set(...)``
        first.

    Subclasses MUST register via ``@register_inference("slug")`` —
    that decorator stamps :data:`SLUG` and adds the class to the
    inference registry. The class also declares its catalog
    metadata (:attr:`LABEL`, :attr:`NEURAL`, :meth:`variants`) so the
    UI can list it without instantiating.
    """

    SLUG: ClassVar[str] = ""        # set by @register_inference decorator
    LABEL: ClassVar[str] = ""       # human-facing display name

    # Optional grouping tag — siblings that share an upstream
    # architecture but ship as separate inference classes (e.g.
    # ``CREStereoInitialInference`` + ``CREStereoCombinedInference``)
    # set this to a common string (``"CREStereo"``) so the UI can
    # group them under one heading. ``None`` for inferences that don't
    # share a family (the default).
    ARCHITECTURE_FAMILY: ClassVar["str | None"] = None

    # When True, ``postprocess`` returns metric-ish depth (HxW float32
    # meters) directly. Pipeline skips disparity filters and the
    # calibration's :meth:`disparity_to_depth` step, routing the value
    # straight into the depth-domain stage. Used by monocular depth
    # backends (Depth Anything, etc.). Stereo backends leave it False
    # and emit disparity in canvas pixels (the existing contract).
    OUTPUTS_DEPTH: ClassVar[bool] = False

    # Whether variants need a downloaded .onnx + engine. False for
    # purely classical inferences (SGBM, StereoBM, rectify-only) —
    # those construct directly with no network/file. The free
    # :func:`build_inference` reads this to decide whether to build an
    # engine before instantiating.
    NEURAL: ClassVar[bool] = True

    # ── Catalog (classmethods) ───────────────────────────────────────

    @classmethod
    def variants(cls) -> "tuple":
        """Static catalog of every supported variant. Pure function —
        no I/O, no environment reads, no caching needed (cheap to
        recompute). Default empty; concrete inferences override.
        ``@register_inference`` wraps this method to stamp the
        inference's slug onto each yielded :class:`Variant`."""
        return ()

    @classmethod
    def get_variant(cls, slug: str):
        """Look up a single variant by slug. Convenience over
        ``next(v for v in cls.variants() if v.slug == slug)``. Raises
        ``KeyError`` with a helpful message listing the inference's
        known variant slugs."""
        for v in cls.variants():
            if v.slug == slug:
                return v
        raise KeyError(
            f"unknown variant {slug!r} in inference {cls.SLUG!r}; "
            f"available: {tuple(v.slug for v in cls.variants())}")

    # ── Properties the pipeline reads ────────────────────────────────

    @property
    @abstractmethod
    def input_size(self) -> tuple[int, int]:
        """``(canvas_h, canvas_w)`` — drives the calibration's canvas
        remap. For neural backends this is the model's compiled input
        size; for SGBM it's whatever the user picked at construction."""

    @property
    def label(self) -> str:
        """Human-readable label for UI / logs (e.g. ``"OpenVINO CPU
        (lightstereo-II)"``)."""
        return self.SLUG

    # ── Hot-path inference API ────────────────────────────────────────

    @abstractmethod
    def submit(self,
               canvas_l: np.ndarray,
               canvas_r: np.ndarray) -> InferenceFuture:
        """Submit one inference. Sync backends wrap the result in a
        pre-completed :class:`EventFuture` so the worker's polling
        loop is uniform."""

    @abstractmethod
    def postprocess(self,
                    raw_output: np.ndarray,
                    canvas_layout: CanvasLayout) -> np.ndarray:
        """Raw inference output → 2D disparity in canvas pixels
        (``valid_h × valid_w`` float32, sliced to the unpadded region
        of the canvas)."""

    # ── Lifecycle ─────────────────────────────────────────────────────

    def warmup(self) -> None:
        """Pre-compile / pre-allocate. Default no-op (classical
        backends usually have nothing to warm); neural backends
        override to compile the engine."""

    def abort(self) -> None:
        """Abort the in-flight inference NOW (stop/cancel path).
        Default no-op — classical backends run synchronously; neural
        backends delegate to the engine (see
        :class:`~depz_camera_sdk.engines.base.AbortedInference`)."""

    def shutdown(self, abandon: bool = False) -> None:
        """Release native resources. Default no-op; override only when
        a backend holds GPU/NPU contexts. ``abandon=True`` (stop path)
        must not block on in-flight work."""

    # ── Camera-aware hook (focus-aware mono backends) ────────────────

    def set_intrinsics(self,
                       fx: float, fy: float,
                       cx: float, cy: float) -> None:
        """Push canvas-space intrinsics to the inference. Default
        no-op — most backends don't need them (classical matchers,
        relative-depth nets, stereo nets that already consume a
        rectified pair).

        Called by the pipeline from
        :meth:`DepthPipeline._rebuild_calibration_geometry` whenever
        the canvas remap is rebuilt (start, calibration update, mode
        change). Focus-aware monocular metric-depth architectures
        (UniDepth, Metric3D, DepthPro) override to remember these
        and use them in their preprocess to produce metric depth."""

    # Runtime controls are introspected via the inherited
    # :meth:`ControlOwner.controls` — the ``_CONTROL_SPECS`` tuple (empty for
    # inferences with no runtime controls: neural, rectify-only).


# ──────────────────────── Neural intermediate bases ──────────────────


class NeuralInferenceBase(DepthInferenceBase):
    """Engine + per-network preprocess/postprocess. Subclassed by
    :class:`NeuralStereoInferenceBase` (stereo, ``OUTPUTS_DEPTH=False``)
    and :class:`NeuralMonoInferenceBase` (mono, ``OUTPUTS_DEPTH=True``).
    Concrete-network classes (under ``architectures/``) subclass one
    of those and override :meth:`preprocess` / :meth:`postprocess`."""

    SLUG: ClassVar[str] = ""    # set by @register_inference decorator
    OUTPUTS_DEPTH: ClassVar[bool] = False

    # Optional override: when set to ``(H, W)``, the engine reshapes
    # the (potentially dynamic-shape) ONNX to this concrete input
    # before compile. Used by networks whose ONNX export ships with
    # dynamic H/W (e.g. Metric3D V2). Leave None to let the engine
    # read the shape from the ONNX directly (the common case).
    INPUT_SIZE: ClassVar["tuple[int, int] | None"] = None

    # Optional: name of the model output the engine should read. Used
    # by multi-output networks (e.g. Metric3D V2 ships
    # ``predicted_depth`` + normals + confidence — we only want the
    # depth head). Leave ``None`` to use the first output, which is
    # correct for every single-head export.
    OUTPUT_NAME: ClassVar["str | None"] = None

    def __init__(self, engine: InferenceEngine) -> None:
        """Bind to a pre-built engine (already loaded with the .onnx
        file). :func:`build_inference` constructs the engine via
        :func:`make_engine`; tests pass mock engines."""
        self._engine: InferenceEngine = engine
        self._init_extra_state()

    def _init_extra_state(self) -> None:
        """Subclass hook — populate any per-instance state the regular
        ``__init__`` would have set (e.g. mono's depth_scale defaults).
        Default: nothing extra."""

    # ── Read-only props ─────────────────────────────────────────────

    @property
    def architecture_slug(self) -> str:
        """Network identity slug (e.g. ``"lightstereo"``). Set by the
        ``@register_inference("...")`` decorator on the concrete
        subclass."""
        return type(self).SLUG

    @property
    def engine_slug(self) -> str:
        """Engine slug (e.g. ``"openvino-CPU"``). Read from the engine
        instance, which :func:`make_engine` stamps at construction."""
        return self._engine.slug

    @property
    def model_path(self) -> str:
        """Resolved absolute path of the .onnx file."""
        return self._engine.model_path

    @property
    def input_size(self) -> tuple[int, int]:
        return self._engine.input_size

    @property
    def label(self) -> str:
        return f"{self.architecture_slug} on {self._engine.slug}"

    # ── Lifecycle ───────────────────────────────────────────────────

    def warmup(self) -> None:
        self._engine.warmup()

    def abort(self) -> None:
        self._engine.abort()

    def shutdown(self, abandon: bool = False) -> None:
        self._engine.shutdown(abandon=abandon)

    def set_intrinsics(self,
                       fx: float, fy: float,
                       cx: float, cy: float) -> None:
        """Optional hook — focus-aware mono networks (e.g. Metric3D
        V2) override to remember the values for use in
        :meth:`preprocess`. Default no-op."""

    # ── Per-network adapter (subclass overrides) ────────────────────

    @abstractmethod
    def preprocess(self,
                   canvas_l: np.ndarray,
                   canvas_r: "np.ndarray | None",
                   ) -> tuple[np.ndarray, ...]:
        """BGR canvas pair (uint8 HxWx3) → engine input blobs.

        Returns NCHW float32 blobs in the order the model declares
        its inputs. Most networks return ``(left_blob, right_blob)``;
        multi-stage ones (CREStereo combined) return more — e.g. 4
        blobs. Mono networks ignore ``canvas_r`` (``None`` on the
        mono pipeline path) and return a single-input tuple."""

    @abstractmethod
    def postprocess(self,
                    raw_output: np.ndarray,
                    canvas_layout: CanvasLayout) -> np.ndarray:
        """Raw engine output → 2D map in canvas pixels
        (``valid_h × valid_w`` float32, sliced to the unpadded region
        of the canvas). Stereo subclasses return disparity; mono
        subclasses return depth (or relative depth)."""

    # ── Inference ───────────────────────────────────────────────────

    def submit(self,
               canvas_l: np.ndarray,
               canvas_r: "np.ndarray | None") -> InferenceFuture:
        blobs = self.preprocess(canvas_l, canvas_r)
        return self._engine.submit(*blobs)


class NeuralStereoInferenceBase(NeuralInferenceBase):
    """Stereo (disparity) inference backed by an ONNX/OpenVINO engine.

    Per-network inferences (:class:`LightStereoInference`,
    :class:`CREStereoInitialInference`, …) subclass this and define
    their own ``preprocess`` / ``postprocess``. Construction takes
    only a pre-built :class:`InferenceEngine`.

    Engine kwargs (forwarded via ``engine_kwargs={...}`` from
    :func:`build_inference`):

    - ``cache_dir`` — override the SDK-wide cache root for compiled-
      model storage. Defaults to :func:`depz_camera_sdk.get_cache_dir`;
      backends that cache (OpenVINO) write into ``<cache_dir>/<backend>/``.
    - ``do_models_cache`` (OpenVINO) — set False to skip caching
      entirely (useful for read-only filesystems / one-shot scripts).
    - ``nireq`` (OpenVINO) — async-queue depth.

    Controls: empty list — there are no runtime-tunable controls once
    the model is compiled. Tune via different weights / different
    engine slug instead.
    """

    SLUG = "neural-stereo"
    OUTPUTS_DEPTH = False


class NeuralMonoInferenceBase(NeuralInferenceBase):
    """Monocular-depth base. Subclasses define the network-specific
    preprocess + postprocess and call :meth:`_apply_depth_affine` at
    the tail of postprocess.

    The two controls ``depth_scale`` and ``depth_offset`` apply an affine
    ``scale·raw + offset``. Defaults (1.0 / 0.0) — set via
    :class:`Control` after construction. Tune them once you
    know one reference distance.
    """

    SLUG = "neural-mono"
    OUTPUTS_DEPTH = True

    # depth_scale / depth_offset — affine ``scale·raw + offset`` applied
    # at the tail of postprocess. Typed control descriptors; subclasses
    # inherit them. ControlOwner seeds the defaults and collects them.
    depth_scale = FloatControl("number", default=1.0, label="Depth scale",
                            min=0.001, max=1000.0, step=0.001)
    depth_offset = FloatControl("number", default=0.0, label="Depth offset",
                             min=-100.0, max=100.0, step=0.01, unit="m")

    def _init_extra_state(self) -> None:
        # Seed the affine controls from their descriptor defaults so
        # postprocess works without an explicit setter call.
        self._seed_control_defaults()

    def _apply_depth_affine(self, depth: np.ndarray) -> np.ndarray:
        """Apply ``scale·depth + offset`` if either control differs from
        default. Subclasses call this at the tail of their
        :meth:`postprocess`."""
        if self.depth_scale != 1.0 or self.depth_offset != 0.0:
            depth = (depth * np.float32(self.depth_scale)
                     + np.float32(self.depth_offset))
            depth = np.ascontiguousarray(depth, dtype=np.float32)
        return depth


# ──────────────────────── Classical intermediate base ────────────────


class ClassicStereoInferenceBase(DepthInferenceBase):
    """Common scaffolding for OpenCV block matchers
    (:class:`SgbmInference`, :class:`StereoBMInference`).

    Subclasses implement :meth:`_rebuild_matcher` (build / re-build
    the cv2 matcher from the current attribute values) and
    :meth:`controls` (the runtime control descriptor list). Everything
    else — fixed canvas size, grayscale conversion, sync future,
    int16-to-pixel cast, postprocess crop, set() rebuild trigger —
    is shared here.
    """

    def __init__(self, canvas_size: "tuple[int, int]") -> None:
        if (len(canvas_size) != 2
                or any(not isinstance(v, int) or v <= 0
                       for v in canvas_size)):
            raise ValueError(
                f"canvas_size must be (H, W) of positive ints; "
                f"got {canvas_size!r}")
        self._canvas_h, self._canvas_w = canvas_size
        # Seed control backing attrs from descriptor defaults; the subclass
        # __init__ calls `_rebuild_matcher()` once super() returns.
        self._seed_control_defaults()
        self._matcher = None    # type: ignore[assignment]

    @property
    def input_size(self) -> tuple[int, int]:
        return self._canvas_h, self._canvas_w

    def submit(self,
               canvas_l: np.ndarray,
               canvas_r: np.ndarray) -> InferenceFuture:
        gray_l = cv2.cvtColor(canvas_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(canvas_r, cv2.COLOR_BGR2GRAY)
        assert self._matcher is not None
        # CV_16S, scaled by 16. We defer the /16 cast to postprocess
        # so the engine-output contract (raw matcher output) matches
        # how neural backends behave.
        raw = self._matcher.compute(gray_l, gray_r)
        fut = EventFuture()
        fut.set_result(raw)
        return fut

    def postprocess(self,
                    raw_output: np.ndarray,
                    canvas_layout: CanvasLayout) -> np.ndarray:
        disp = raw_output.astype(np.float32, copy=False) / 16.0
        if disp.ndim != 2:
            raise RuntimeError(
                f"{type(self).__name__} output must be 2D, "
                f"got shape {disp.shape}")
        y, x = canvas_layout.pad_y, canvas_layout.pad_x
        h, w = canvas_layout.valid_h, canvas_layout.valid_w
        return np.ascontiguousarray(disp[y: y + h, x: x + w])

    def control_set(self, name, value):
        # Default coerce + setattr, then rebuild matcher so the next
        # submit() picks up the new control. Pipeline funnels control
        # writes through its command queue, so the worker is the only
        # caller — no extra locking needed.
        applied = super().control_set(name, value)
        self._rebuild_matcher()
        return applied

    def _rebuild_matcher(self) -> None:
        """Build / rebuild the cv2 matcher from the current attribute
        values. Called once at the end of __init__ and after every
        control mutation. Subclass-specific."""
        raise NotImplementedError
