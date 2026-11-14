"""All inference base classes — public ABCs + intermediate bases.

This module hosts every reusable base. Concrete inferences live in
``inferences/<network>.py``, one file per network.

Hierarchy::

    DepthInferenceBase                 ← public ABC, pipeline-facing,
    │                                    catalog metadata (LABEL,
    │                                    NEURAL, variants(), …)
    ├── NeuralInferenceBase            ← engine + input normalisation
    │   │                                (BLOB_MEAN / BLOB_SCALE) +
    │   │                                output → 2-D map extraction
    │   ├── NeuralStereoInferenceBase  ← two-eye preprocess, disparity out
    │   └── NeuralMonoInferenceBase    ← one-eye preprocess, depth out,
    │                                     depth_scale/offset controls
    ├── ClassicStereoInferenceBase     ← cv2-matcher scaffold
    └── (RectifyOnlyInference, lives in inferences/rectify_only.py)

The descriptor (:class:`Control`) and the get/set/coerce
machinery are shared with :class:`~depz_camera_sdk.filters.base.Filter`
via :class:`~depz_camera_sdk.._controls.ControlOwner`."""

import dataclasses
import hashlib
import json
import logging
import re
import sys
import tempfile
import time
import urllib.error
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
from ._download import (
    DEFAULT_TIMEOUT_S as _DOWNLOAD_TIMEOUT_S,
    HTTP_USER_AGENT as _HTTP_USER_AGENT,
    DownloadError,
    FileLock,
    http_download,
    remove_stale_parts,
    unique_temp_path,
)

__all__ = [
    "Control",
    "DepthInferenceBase",
    "NeuralStereoInferenceBase",
    "NeuralMonoInferenceBase",
    "InferenceVariant",
    "ModelFile",
    "onnx_variant",
    "build_inference",
    "stderr_progress",
    "IMAGENET_BLOB_MEAN",
    "IMAGENET_BLOB_SCALE",
]


_LOGGER = logging.getLogger("depz_camera_sdk.inference")


# ───────────────────────── Slug → URL resolver ────────────────────────
# The catalog endpoint + per-process slug cache. ``ModelFile._resolve_url``
# reads/writes ``_slug_cache`` so a slug only round-trips once per
# session. Tests patch :data:`_SLUG_RESOLVER_URL` (or seed
# ``_slug_cache`` directly) to bypass the network.


_SLUG_RESOLVER_URL = "https://depz.ai/api/public/get-model-file-link"


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


# ───────────────────────── Input normalisation ────────────────────────
# A network input blob is ``(RGB - mean) * scale`` per channel, NCHW
# float32 — one ``cv2.dnn.blobFromImageWithParams`` call. The two
# recipes every catalog model uses:

# ImageNet statistics on the 0..255 scale: (img/255 - mean) / std.
IMAGENET_BLOB_MEAN = (0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0, 0.0)
IMAGENET_BLOB_SCALE = (
    1.0 / (255.0 * 0.229),
    1.0 / (255.0 * 0.224),
    1.0 / (255.0 * 0.225),
    0.0,
)

# Raw RGB in [0, 255] — for exports that normalise inside the graph.
_RAW_BLOB_MEAN = (0.0, 0.0, 0.0, 0.0)
_RAW_BLOB_SCALE = (1.0, 1.0, 1.0, 0.0)


def _meta_path(target: Path) -> Path:
    """Sidecar written next to every file the SDK downloads."""
    return target.with_name(target.name + ".meta.json")


def _read_meta(target: Path) -> Optional[dict]:
    try:
        meta = json.loads(_meta_path(target).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return meta if isinstance(meta, dict) else None


def _write_meta(target: Path, *, size: int, sha256: str, slug: str) -> None:
    path = _meta_path(target)
    tmp = unique_temp_path(path)
    try:
        tmp.write_text(json.dumps(
            {"size": size, "sha256": sha256, "slug": slug}), encoding="utf-8")
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def stderr_progress() -> Callable[[int, int], None]:
    """A ``download_progress_cb`` that draws a one-line progress display
    on stderr — for scripts and CLIs; the library itself never prints.
    Throttled to ~10 Hz; ends the line when the download completes.
    Make a fresh one per :func:`build_inference` /
    :meth:`InferenceVariant.resolve_files` call::

        build_inference(slug, variant, download_progress_cb=stderr_progress())
    """
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

    ``sha256`` and ``size_bytes`` are optional declarations of what the
    stored file must be; both are verified when the file is downloaded
    (a mismatch fails the download and caches nothing). ``size_bytes``
    also gives the UI a progress total before the HTTP response is read.

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
        """True iff ``target`` is on disk and passes the cheap integrity
        checks. Read-only — never opens a socket, never hashes.

        Checked: the file exists; its size equals :attr:`size_bytes`
        (when declared); and, when the SDK's ``<file>.meta.json`` sidecar
        is present, the size equals the one recorded at download time
        and the sidecar was written for this same :attr:`slug`. A file
        without a sidecar (placed by hand, or cached by an older SDK) is
        accepted on the size check alone.

        sha256 is NOT recomputed here — it is verified once, at download
        time, before the file enters the cache."""
        try:
            st = target.stat()
        except OSError:
            return False
        if not target.is_file():
            return False
        if self.size_bytes is not None and st.st_size != self.size_bytes:
            return False
        meta = _read_meta(target)
        if meta is not None:
            if meta.get("size") != st.st_size:
                return False
            if meta.get("slug") not in (None, self.slug):
                return False
        return True

    def resolve(
            self,
            target: Path,
            download_progress_cb: "Optional[Callable[[int, int], None]]" = None,
    ) -> None:
        """Ensure the file is on disk at ``target``: no-op if
        :meth:`is_cached`, otherwise download it. Safe to call from
        several processes at once — one downloads, the others wait and
        then find the file cached."""
        if self.is_cached(target):
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(self._lock_path(target)):
            if self.is_cached(target):
                return      # another process fetched it while we waited
            self._download_locked(target, download_progress_cb,
                                  _DOWNLOAD_TIMEOUT_S)

    @staticmethod
    def _lock_path(target: Path) -> Path:
        return target.with_name(target.name + ".lock")

    def _resolve_url(self, *, timeout: float = 10.0) -> str:
        """Resolve :attr:`slug` to its public download URL via the SDK
        catalog service. Cached per process in :data:`_slug_cache`."""
        cached = _slug_cache.get(self.slug)
        if cached is not None:
            return cached
        qs = urllib.parse.quote(self.slug, safe="")
        url = f"{_SLUG_RESOLVER_URL}?slug={qs}"
        req = urllib.request.Request(url, headers={"User-Agent": _HTTP_USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"failed to resolve slug {self.slug!r}: "
                f"HTTP {e.code} {e.reason}") from e
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
            *,
            timeout: float = _DOWNLOAD_TIMEOUT_S,
    ) -> None:
        """Fetch the file and land it at ``target`` atomically, replacing
        whatever is there. Direct download by default; archive-extract
        flow when :attr:`archive_member` is set.

        The stored file is verified against :attr:`size_bytes` and
        :attr:`sha256` (when declared) and, for direct downloads,
        against the server's Content-Length; on any mismatch nothing is
        cached and ``RuntimeError`` is raised. ``timeout`` bounds each
        connect / read, not the whole transfer. Concurrent downloads of
        the same target — other threads or other processes — are
        serialized by a lock file next to it."""
        target.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(self._lock_path(target)):
            self._download_locked(target, download_progress_cb, timeout)

    def _download_locked(
            self, target: Path,
            progress_cb: "Optional[Callable[[int, int], None]]",
            timeout: float) -> None:
        remove_stale_parts(target)
        tmp = unique_temp_path(target)
        try:
            if self.archive_member is None:
                size, digest = self._fetch(tmp, progress_cb, timeout)
            else:
                with tempfile.TemporaryDirectory(
                        prefix="depz-archive-", dir=target.parent) as td:
                    arch_path = Path(td) / "archive"
                    self._fetch(arch_path, progress_cb, timeout)
                    size, digest = self._extract_member(arch_path, tmp)
            if self.size_bytes is not None and size != self.size_bytes:
                raise RuntimeError(
                    f"size mismatch for slug {self.slug!r}: expected "
                    f"{self.size_bytes} bytes, got {size}")
            if self.sha256 is not None and digest != self.sha256:
                raise RuntimeError(
                    f"sha256 mismatch for slug {self.slug!r}: "
                    f"expected {self.sha256}, got {digest}")
            # Drop the old sidecar first: if the process dies between the
            # two replaces, the new file is merely "no sidecar" (accepted
            # on size) rather than contradicted by a stale one.
            _meta_path(target).unlink(missing_ok=True)
            tmp.replace(target)
            _write_meta(target, size=size, sha256=digest, slug=self.slug)
        finally:
            tmp.unlink(missing_ok=True)

    def _fetch(self, dest: Path,
               progress_cb: "Optional[Callable[[int, int], None]]",
               timeout: float) -> tuple[int, str]:
        """Resolve the slug and stream the payload into ``dest``;
        returns ``(size, sha256)`` of what was written. A link taken
        from the per-process cache that answers 403/404 (expired signed
        URL, file moved) is re-resolved once."""
        link_was_cached = self.slug in _slug_cache
        url = _direct_download_url(self._resolve_url())
        _LOGGER.info("downloading %s (slug=%s) → %s", url, self.slug, dest)
        try:
            try:
                return http_download(
                    url, dest, progress_cb=progress_cb,
                    expected_size=self.size_bytes, timeout=timeout)
            except DownloadError as e:
                if not (link_was_cached and e.status in (403, 404)):
                    raise
                _LOGGER.info(
                    "cached link for slug %s answered HTTP %s — resolving "
                    "it again", self.slug, e.status)
                _slug_cache.pop(self.slug, None)
                url = _direct_download_url(self._resolve_url())
                return http_download(
                    url, dest, progress_cb=progress_cb,
                    expected_size=self.size_bytes, timeout=timeout)
        except DownloadError as e:
            raise RuntimeError(
                f"download of slug {self.slug!r} failed: {e}") from e

    def _extract_member(self, archive: Path, dest: Path) -> tuple[int, str]:
        """Copy :attr:`archive_member` out of the zip at ``archive`` into
        ``dest``; returns ``(size, sha256)`` of the extracted file. The
        member is matched by full path first, then by basename."""
        member = self.archive_member
        with zipfile.ZipFile(archive) as zf:
            try:
                info = zf.getinfo(member)
            except KeyError:
                info = next(
                    (i for i in zf.infolist()
                     if Path(i.filename).name == member),
                    None)
                if info is None:
                    raise RuntimeError(
                        f"archive for slug {self.slug!r} has no member "
                        f"{member!r}; available: "
                        f"{[i.filename for i in zf.infolist() if not i.is_dir()]}"
                    ) from None
            digest = hashlib.sha256()
            size = 0
            with zf.open(info) as src, dest.open("wb") as dst:
                for chunk in iter(lambda: src.read(1 << 20), b""):
                    dst.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
        return size, digest.hexdigest()


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

    ``input_size`` is the ``(H, W)`` canvas the variant runs at — the
    model's fixed input size for neural variants, the chosen working
    size for classical ones. ``None`` when a variant does not pin one.

    ``inference_kwargs`` are forwarded verbatim to the inference's
    constructor by :func:`build_inference`
    (``cls(engine=..., **variant.inference_kwargs)``). Use this for
    per-variant ctor args — SGBM/StereoBM/RectifyOnly ``canvas_size``,
    etc.

    ``configurations`` is opaque to the framework — free-form metadata
    for consumers; the catalog stores the viewer's model-picker facts
    under ``"selector_meta"``. Not passed to the constructor or the
    engine.

    ``inference_slug`` is bound by :meth:`DepthInferenceBase.variants`
    when the inference enumerates its catalog — concrete classes don't
    pass it when constructing the dataclass."""

    slug: str
    label: str
    files_to_download: tuple[ModelFile, ...] = ()
    input_size: Optional[tuple[int, int]] = None
    inference_kwargs: dict[str, Any] = field(default_factory=dict)
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
        """True iff every file in :attr:`files_to_download` passes
        :meth:`ModelFile.is_cached` (present, size consistent). Cheap:
        ``stat()`` calls and small sidecar reads only."""
        cache_dir = self.cache_dir
        return all(f.is_cached(cache_dir / f.filename)
                   for f in self.files_to_download)

    def resolve_files(
            self, *,
            download_progress_cb: "Optional[Callable[[int, int], None]]" = None,
    ) -> list[Path]:
        """Ensure the variant's cache directory is populated, then
        return absolute paths of every file in
        :attr:`files_to_download` (entry-point first).

        Silent by default. ``download_progress_cb(done_bytes,
        total_bytes)`` is called while a file is being transferred
        (never on a cache hit); pass :func:`stderr_progress` for a
        ready-made console progress line."""
        cache_dir = self.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        for f in self.files_to_download:
            f.resolve(cache_dir / f.filename, download_progress_cb)
        return [cache_dir / f.filename for f in self.files_to_download]


def onnx_variant(*, slug: str, label: str, input_size: tuple[int, int],
                 model_slug: str, selector_meta: dict,
                 archive_member: Optional[str] = None,
                 sha256: Optional[str] = None,
                 size_bytes: Optional[int] = None) -> InferenceVariant:
    """The common catalog entry: one ONNX file, stored as ``model.onnx``
    in the variant's cache directory, running at a fixed ``input_size``
    ``(H, W)``. ``model_slug`` is the model-registry identifier of the
    download (see :class:`ModelFile`); ``archive_member`` names the
    ``.onnx`` inside it when the download is a zip. ``selector_meta``
    lands in ``configurations["selector_meta"]``."""
    return InferenceVariant(
        slug=slug,
        label=label,
        input_size=(int(input_size[0]), int(input_size[1])),
        files_to_download=(
            ModelFile(slug=model_slug, filename="model.onnx",
                      archive_member=archive_member,
                      sha256=sha256, size_bytes=size_bytes),
        ),
        configurations={"selector_meta": selector_meta},
    )


def build_inference(
    inference_slug: str,
    inference_variant_slug: str,
    *,
    engine_slug: "Optional[str]" = None,
    download_progress_cb: "Optional[Callable[[int, int], None]]" = None,
) -> "DepthInferenceBase":
    """One-shot: ``(inference_slug, variant_slug)`` → fully-initialized
    :class:`DepthInferenceBase`.

    1. :func:`get_inference` resolves the inference class, then
       :meth:`get_variant` looks up the variant.
    2. :meth:`InferenceVariant.resolve_files` populates the cache
       (downloads on first use; cache hit is read-only ``stat()``s).
    3. Classical inferences (``cls.NEURAL = False``) are constructed
       directly with ``variant.inference_kwargs``. Neural ones get an
       :class:`InferenceEngine` built around the entry-point file and
       passed via ``engine=``.

    ``engine_slug`` defaults to :func:`recommend_engine` (best engine
    available on this host); on an NVIDIA GPU with TensorRT installed
    that default becomes ``tensorrt-fp16`` / ``-fp32`` for the models
    verified for it (:func:`depz_camera_sdk.engines.engine_for_model`).
    An explicit ``engine_slug`` is used as given. Downloads are silent unless
    ``download_progress_cb(done_bytes, total_bytes)`` is given —
    :func:`stderr_progress` returns one that draws a console progress
    line.

    To pass engine options (``cache_dir``, ``nireq``, …) build the
    engine yourself: ``cls(engine=make_engine(slug, path, **options))``.
    """
    from .registry import get_inference
    cls = get_inference(inference_slug)
    variant = cls.get_variant(inference_variant_slug)

    # Pick the engine before fetching anything: without one, a neural
    # model's download (up to ~150 MB) would be for nothing.
    if cls.NEURAL and engine_slug is None:
        from ..engines import engine_for_model, recommend_engine
        engine_slug = engine_for_model(
            cls.SLUG, inference_variant_slug, recommend_engine())
        if engine_slug is None:
            raise RuntimeError(
                f"no inference engine is installed, and {cls.SLUG!r} is a "
                f"neural model that needs one. The base depz-camera-sdk "
                f"install has none — add an extra, e.g. "
                f"`pip install \"depz-camera-sdk[openvino]\"` (Intel/AMD "
                f"x86_64) or `\"depz-camera-sdk[cpu]\"` (ONNX Runtime; "
                f"also [cuda12-bundled], [cuda13-bundled], [dml]). The classical "
                f"models (sgbm, stereo-bm) run without an engine.")

    files = variant.resolve_files(download_progress_cb=download_progress_cb)

    if not cls.NEURAL:
        return cls(**variant.inference_kwargs)

    if not files:
        raise RuntimeError(
            f"variant {variant.slug!r} is neural ({cls.SLUG!r}) but "
            f"declares no files_to_download — no .onnx for the engine")

    # Inferences whose ONNX is dynamic-shape declare INPUT_SIZE so the
    # engine reshapes to a concrete (1, 3, H, W) before compile.
    # Multi-head exports (Metric3D V2 → predicted_depth + normals +
    # confidence) declare OUTPUT_NAME so the engine binds to the right
    # output instead of the default first-output index.
    from ..engines import make_engine
    engine_options: dict[str, Any] = {}
    if cls.INPUT_SIZE is not None:
        h, w = cls.INPUT_SIZE
        engine_options["input_shape"] = (1, 3, h, w)
    if cls.OUTPUT_NAME is not None:
        engine_options["output_name"] = cls.OUTPUT_NAME
    engine = make_engine(engine_slug, str(files[0]), **engine_options)

    return cls(engine=engine, **variant.inference_kwargs)


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

    Concrete classes register via ``@register_inference("slug")`` —
    the decorator sets :data:`SLUG` and adds the class to the inference
    registry. The class declares its catalog metadata (:attr:`LABEL`,
    :attr:`NEURAL`, and its variants via :meth:`_variants`) so the UI
    can list it without instantiating.
    """

    SLUG: ClassVar[str] = ""        # set by @register_inference decorator
    LABEL: ClassVar[str] = ""       # human-facing display name

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
    def variants(cls) -> "tuple[InferenceVariant, ...]":
        """Catalog of every supported variant, each bound to this
        inference's :data:`SLUG` (the cache layout, the download flow
        and :func:`build_inference` need that back-pointer). Pure — no
        I/O, cheap to recompute.

        Do not override: declare the variants in :meth:`_variants`."""
        slug = cls.SLUG
        return tuple(
            v if v._inference_slug == slug
            else dataclasses.replace(v, _inference_slug=slug)
            for v in cls._variants())

    @classmethod
    def _variants(cls) -> "tuple[InferenceVariant, ...]":
        """The variants this inference offers, in display order (the
        first one is what "first variant" fallbacks pick). Concrete
        classes override; build each :class:`InferenceVariant` without
        an inference slug — :meth:`variants` binds it."""
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

        Called by the pipeline whenever the canvas remap is rebuilt
        (start, calibration update, fit-mode change). Focus-aware
        monocular metric-depth networks (Metric3D) override to remember
        them and scale their output to metric depth."""

    # Runtime controls are introspected via the inherited
    # :meth:`ControlOwner.controls` (empty for inferences with no
    # runtime controls: neural stereo, rectify-only).


# ──────────────────────── Neural intermediate bases ──────────────────


class NeuralInferenceBase(DepthInferenceBase):
    """Engine + the pre/post-processing every network shares. Subclassed
    by :class:`NeuralStereoInferenceBase` (two eyes in, disparity out)
    and :class:`NeuralMonoInferenceBase` (one eye in, depth out), which
    supply default :meth:`preprocess` / :meth:`postprocess`. A concrete
    network class (under ``inferences/``) normally only sets class
    attributes — :attr:`BLOB_MEAN` / :attr:`BLOB_SCALE`,
    :attr:`OUTPUT_CHANNEL`, :attr:`INPUT_SIZE`, :attr:`OUTPUT_NAME` —
    and overrides a method only where its network really differs."""

    OUTPUTS_DEPTH: ClassVar[bool] = False

    # Input normalisation: blob = (RGB - BLOB_MEAN) * BLOB_SCALE per
    # channel (4-tuples: R, G, B, unused). The default feeds raw RGB in
    # [0, 255], for exports that normalise inside the graph; set both to
    # IMAGENET_BLOB_MEAN / IMAGENET_BLOB_SCALE for exports that expect
    # ImageNet-normalised input.
    BLOB_MEAN: ClassVar[tuple] = _RAW_BLOB_MEAN
    BLOB_SCALE: ClassVar[tuple] = _RAW_BLOB_SCALE

    # Which channel of a ``(1, C, H, W)`` output is the map. ``None``
    # (default): the output has a single map and must squeeze to 2-D.
    # Set for networks that emit more (CREStereo: channel 0 of a 2-channel
    # flow field is the horizontal disparity).
    OUTPUT_CHANNEL: ClassVar["int | None"] = None

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
        self.blob_params = self.make_blob_params()
        """The ``cv2.dnn.Image2BlobParams`` this instance converts a BGR
        canvas with — see :meth:`make_blob_params`."""
        self._init_extra_state()

    @classmethod
    def make_blob_params(cls) -> "cv2.dnn.Image2BlobParams":
        """The network's input recipe as ``cv2.dnn.Image2BlobParams``:
        BGR uint8 HxWx3 → RGB, ``(x - BLOB_MEAN) * BLOB_SCALE``, NCHW
        float32, no resize. A classmethod so tools can preprocess for a
        network without building an engine:
        ``cv2.dnn.blobFromImageWithParams(img, cls.make_blob_params())``."""
        return cv2.dnn.Image2BlobParams(
            scalefactor=cls.BLOB_SCALE,
            size=(0, 0),
            mean=cls.BLOB_MEAN,
            swapRB=True,
            ddepth=cv2.CV_32F,
        )

    def _init_extra_state(self) -> None:
        """Subclass hook, called at the end of ``__init__``: set up any
        extra per-instance state (mono's control defaults, Metric3D's
        focal bookkeeping). Default: nothing."""

    def _to_blob(self, canvas: np.ndarray) -> np.ndarray:
        """One BGR canvas → the network's NCHW float32 input blob."""
        return cv2.dnn.blobFromImageWithParams(canvas, self.blob_params)

    def _extract_map(self, raw_output: np.ndarray,
                     canvas_layout: CanvasLayout) -> np.ndarray:
        """Raw engine output → the 2-D float32 map of the canvas's valid
        region: pick :attr:`OUTPUT_CHANNEL` (or squeeze), check it is
        2-D, cast, crop the letterbox padding away."""
        arr = np.asarray(raw_output)
        channel = self.OUTPUT_CHANNEL
        if channel is None:
            out = np.squeeze(arr)
            if out.ndim != 2:
                raise RuntimeError(
                    f"model output shape {tuple(arr.shape)} does not yield "
                    f"a 2D map after squeeze (got {tuple(out.shape)})")
        else:
            if arr.ndim != 4 or arr.shape[0] != 1 or arr.shape[1] <= channel:
                raise RuntimeError(
                    f"{self.SLUG} expects raw output shape (1, C, H, W) "
                    f"with C > {channel}; got {tuple(arr.shape)}")
            out = arr[0, channel]
        if out.dtype != np.float32:
            out = out.astype(np.float32, copy=False)
        y, x = canvas_layout.pad_y, canvas_layout.pad_x
        h, w = canvas_layout.valid_h, canvas_layout.valid_w
        return np.ascontiguousarray(out[y: y + h, x: x + w])

    # ── Read-only props ─────────────────────────────────────────────

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
        return f"{self.SLUG} on {self._engine.slug}"

    # ── Lifecycle ───────────────────────────────────────────────────

    def warmup(self) -> None:
        self._engine.warmup()

    def abort(self) -> None:
        self._engine.abort()

    def shutdown(self, abandon: bool = False) -> None:
        self._engine.shutdown(abandon=abandon)

    # ── Per-network adapter ─────────────────────────────────────────

    @abstractmethod
    def preprocess(self,
                   canvas_l: np.ndarray,
                   canvas_r: "np.ndarray | None",
                   ) -> tuple[np.ndarray, ...]:
        """BGR canvas pair (uint8 HxWx3) → engine input blobs.

        Returns NCHW float32 blobs in the order the model declares
        its inputs. The stereo / mono bases implement the usual
        ``(left, right)`` / ``(left,)``; multi-stage networks (CREStereo
        combined: 4 blobs) override."""

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

    Defaults: both eyes go through the class's blob recipe
    (:attr:`BLOB_MEAN` / :attr:`BLOB_SCALE`) into a two-input network;
    the output map, cropped to the valid canvas region, is the
    disparity in canvas pixels. Per-network classes
    (:class:`LightStereoInference`, :class:`CREStereoInitialInference`,
    …) set class attributes and override only what differs.

    Controls: none — nothing is tunable once the model is compiled.
    Pick different weights or a different engine instead.
    """

    OUTPUTS_DEPTH = False

    def preprocess(self,
                   canvas_l: np.ndarray,
                   canvas_r: "np.ndarray | None",
                   ) -> tuple[np.ndarray, ...]:
        return self._to_blob(canvas_l), self._to_blob(canvas_r)

    def postprocess(self,
                    raw_output: np.ndarray,
                    canvas_layout: CanvasLayout) -> np.ndarray:
        return self._extract_map(raw_output, canvas_layout)


class NeuralMonoInferenceBase(NeuralInferenceBase):
    """Monocular-depth base.

    Defaults: the left eye goes through the class's blob recipe into a
    single-input network (the right canvas is ignored — ``None`` on the
    mono pipeline path); the output map, cropped to the valid canvas
    region and passed through the depth affine, is the depth.

    The two controls ``depth_scale`` and ``depth_offset`` apply
    ``scale·raw + offset`` (defaults 1.0 / 0.0 — a no-op). Tune them
    once you know one reference distance.
    """

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

    def preprocess(self,
                   canvas_l: np.ndarray,
                   canvas_r: "np.ndarray | None",
                   ) -> tuple[np.ndarray, ...]:
        return (self._to_blob(canvas_l),)

    def postprocess(self,
                    raw_output: np.ndarray,
                    canvas_layout: CanvasLayout) -> np.ndarray:
        return self._apply_depth_affine(
            self._extract_map(raw_output, canvas_layout))

    def _apply_depth_affine(self, depth: np.ndarray) -> np.ndarray:
        """Apply ``scale·depth + offset`` if either control differs from
        default."""
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
