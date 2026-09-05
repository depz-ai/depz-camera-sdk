"""Factory + enumeration of available inference engines.

Engines are identified by a **slug string** like `"openvino-CPU"` or
`"onnx-cuda"` — `f"{kind}-{device or ep}"`. There's no descriptor
class; the slug is the identity.

Public API:
    list_engines(scope=...)  → list[str] — one entry point for every
                               engine sublist ("available" / "detected"
                               / "full" / "hidden"), sorted by priority
    available_engines()      → list_engines("available") — kept as the
                               short name for the overwhelmingly common
                               case
    recommend_engine()       → str | None — best one for this host
    make_engine(slug, model) → construct + return InferenceEngine
    parse_engine_slug(slug)  → (kind, device_or_ep) — small helper
                               for UIs that want the parts split.
    priority_rank(slug)      → int — canonical sort key (lower = better)

This module deliberately ships as plain .py in compiled wheels (see
``_compile_manifest``): the tables below are *policy* — which engines
exist, which are surfaced, in what order — and staying readable/patchable
source means a field install can be adjusted without a Cython rebuild.
"""

import logging
import platform
from typing import Iterator, Optional

from .base import InferenceEngine


_LOGGER = logging.getLogger("depz_camera_sdk.engines.registry")


# ── Priority ranking ───────────────────────────────────────────────────
#
# On ARM (aarch64/arm64/armv7l) we prefer onnx-cpu over openvino-CPU:
# OpenVINO's ARM CPU plugin is feature-limited (see the supported-devices
# matrix — CPU_ARM has ~80% API coverage vs ~98% on x86-64), so
# onnxruntime-cpu is the primary ARM path. In practice ARM installs
# don't have OpenVINO at all — the ``[openvino]`` extra is marker-gated
# to Win/Linux x86_64 and the installer never requests it on ARM — but
# the swap stays as a guard for hand-built envs. On Apple-Silicon ARM
# it is moot anyway — onnx-coreml (rank 2) outranks both. It only
# meaningfully affects Linux ARM boxes, where there's no CoreML.
_IS_ARM = platform.machine().lower() in (
    "aarch64", "arm64", "armv7l", "armv8l", "arm")
#
# Engines listed best-first. Every (kind, device-or-ep) combination the
# SDK supports must appear here — :func:`available_engines` raises if
# it discovers anything not in this table, so new devices / EPs fail
# loudly until added.
#
# Heuristic: hardware-accelerated > CPU; vendor-native (TensorRT on
# NVIDIA, CoreML on Apple, OpenVINO on Intel hardware) > generic ONNX.
# Within a kind, GPU/NPU > CPU.
#
# Case convention: each tail uses the case the backend actually
# advertises — OpenVINO `core.available_devices` returns uppercase
# ("CPU", "GPU", "NPU"); ONNX Runtime EPs come back lowercase ("cuda",
# "tensorrt", "dml") via :func:`_short_ep`. Looks mixed but matches
# canonical form, so equality lookups don't need normalization on the
# discovered side. User input is normalized separately in
# :func:`make_engine` (OV uppercase, ONNX lowercase).

_PRIORITY_ORDER: list[tuple[str, str]] = [
    # CUDA before TensorRT: the onnxruntime-gpu wheel bundles its CUDA
    # dependencies on Windows, so `onnx-cuda` is the "just works" NVIDIA
    # path. TensorRT EP is compiled into the same wheel but needs the
    # TensorRT runtime (nvinfer.dll + friends) installed *separately*
    # system-wide; without it the EP loads, silently drops itself from
    # the active session, and inference falls back to CPU — the exact
    # "I picked NVIDIA but it ran on CPU" trap. CUDA-first means the
    # priority winner (recommend_engine / preferred device) works out of
    # the box for everyone with onnxruntime-gpu. Users who have TensorRT
    # set up can still pick it explicitly.
    ("onnx", "cuda"),           # NVIDIA, reliable
    ("onnx", "tensorrt"),       # NVIDIA; hidden from enumeration (see
                                # _HIDDEN_ENGINES) — explicit opt-in only
    ("onnx", "coreml"),         # Apple-native
    ("openvino", "NPU"),        # Intel NPU
    ("openvino", "GPU"),        # Intel GPU
    ("onnx", "dml"),            # Windows generic GPU (DirectML)
    # MIGraphX before raw ROCm — matches the installer's preference
    # (_select_ep returns "migraphx" when _has_migraphx_env() is true).
    # Without this entry onnx-migraphx fell through _priority_rank to
    # rank len(_PRIORITY_ORDER), sorting BELOW onnx-cpu, so recommend_engine()
    # ignored the MIGraphX EP the installer had just picked.
    ("onnx", "migraphx"),       # AMD GPU, MIGraphX (preferred over raw ROCm)
    ("onnx", "rocm"),           # AMD GPU, raw ROCm
    # QNN above both CPU entries: on Windows ARM the QNN EP drives the
    # Qualcomm NPU (and covers the CPU too), so when the onnxruntime-qnn
    # wheel is installed it is strictly better than plain CPU inference.
    # Harmless elsewhere — the EP only exists in that wheel.
    ("onnx", "qnn"),            # Qualcomm NPU (Windows ARM)
    ("openvino", "CPU"),        # Intel CPU
    ("onnx", "cpu"),            # generic CPU fallback
]


# ── Hidden engines ─────────────────────────────────────────────────────
#
# Engine slugs we never surface through enumeration, even when the
# backend advertises them. Hidden ≠ unsupported: `make_engine()` still
# constructs these on explicit request (it validates against the
# backend's raw capability list, not this table) — hiding only removes
# them from `list_engines("available")` and therefore from every
# dropdown, recommendation, and auto-selection built on it.
#
# - "onnx-azure": AzureExecutionProvider is a remote-inference shim
#   (Azure ML endpoints), not a local accelerator. It ships in default
#   onnxruntime wheels but is useless for our offline pipeline and
#   pollutes the engine dropdown.
# - "onnx-tensorrt": onnxruntime-gpu compiles the TensorRT EP in and
#   advertises it via get_available_providers() even when the TensorRT
#   runtime (libnvinfer) is NOT installed — the wheel never bundles it.
#   Selecting it then fails at session creation with a silent CPU
#   fallback (caught by OnnxEngine's guard), and the failed provider
#   load can leave ORT's provider bridge half-initialized so a
#   follow-up session creation segfaults the process. onnx-cuda covers
#   NVIDIA acceleration reliably; users with a working TensorRT setup
#   can still opt in explicitly via make_engine("onnx-tensorrt", ...).
_HIDDEN_ENGINES: frozenset[str] = frozenset({
    "onnx-azure",
    "onnx-tensorrt",
})


# ── Engine classes ─────────────────────────────────────────────────────
#
# Coarse hardware classes for the model wizard: model metadata declares
# requirements as a class ("this variant needs at least igpu",
# "recommended on nvidia"), not as engine slugs, so hosts and models can
# be matched without either knowing the other's specifics. The mapping
# from slug to class is engine knowledge and lives here next to
# _PRIORITY_ORDER / _HIDDEN_ENGINES; the viewer's model_selector (and,
# via the state payload, the web frontend) consume it.

ENGINE_CLASSES: tuple[str, ...] = ("cpu", "igpu", "nvidia")  # worst→best

_ENGINE_CLASS: dict[str, str] = {
    # Hidden engines classify normally — hiding is about enumeration,
    # not capability (an explicit onnx-tensorrt opt-in is still nvidia).
    "onnx-tensorrt":  "nvidia",
    "onnx-cuda":      "nvidia",
    "onnx-coreml":    "igpu",
    "openvino-NPU":   "igpu",
    "openvino-GPU":   "igpu",
    "onnx-dml":       "igpu",
    "onnx-migraphx":  "igpu",
    "onnx-rocm":      "igpu",
    "onnx-qnn":       "igpu",
    "openvino-CPU":   "cpu",
    "onnx-cpu":       "cpu",
}


def classify_engine(slug: str) -> str:
    """Coarse hardware class (see :data:`ENGINE_CLASSES`) for an engine
    slug. OpenVINO multi-device suffixes (``GPU.0``) strip before
    lookup; unknown slugs default to ``"cpu"`` — the conservative floor
    for capability matching."""
    base = slug.split(".", 1)[0]
    return _ENGINE_CLASS.get(base, "cpu")


def class_rank(klass: str) -> int:
    """Sort key for engine classes — higher = more capable. Raises
    :class:`ValueError` on a class not in :data:`ENGINE_CLASSES` (bad
    model metadata should fail loudly, not sort arbitrarily)."""
    return ENGINE_CLASSES.index(klass)


def parse_engine_slug(slug: str) -> tuple[str, str]:
    """Split an engine slug into (kind, device_or_ep). The kind is
    lowercased — user input is accepted in any case; the tail is
    returned verbatim and normalized later by :func:`make_engine`
    per-kind. Raises :class:`ValueError` if the slug doesn't have
    the expected `kind-tail` shape.

    >>> parse_engine_slug("openvino-CPU")
    ('openvino', 'CPU')
    >>> parse_engine_slug("onnx-cuda")
    ('onnx', 'cuda')
    >>> parse_engine_slug("OPENVINO-CPU")
    ('openvino', 'CPU')
    >>> parse_engine_slug("bogus")
    Traceback (most recent call last):
        ...
    ValueError: engine slug must look like '<kind>-<device_or_ep>', got 'bogus'
    """
    if "-" not in slug:
        raise ValueError(
            f"engine slug must look like '<kind>-<device_or_ep>', "
            f"got {slug!r}")
    kind, _, tail = slug.partition("-")
    return kind.lower(), tail


def _priority_rank(slug: str) -> int:
    """Sort key for :func:`available_engines` — lower = higher priority.

    OpenVINO enumerates multiple same-class accelerators as ``GPU.0``,
    ``GPU.1``, … ; :data:`_PRIORITY_ORDER` is keyed by the base class
    (``GPU``), so the ``.N`` suffix is stripped before lookup. A
    combination absent from the table sorts last rather than raising —
    an unrecognised backend stays usable, just deprioritised.

    On ARM, ``onnx-cpu`` outranks ``openvino-CPU`` (see :data:`_IS_ARM`)
    — the two ranks are swapped so the recommended ARM engine is
    ``onnx-cpu``, matching the installer's "ARM CPU → onnx" routing."""
    kind, tail = parse_engine_slug(slug)
    base = tail.split(".")[0]
    try:
        rank = _PRIORITY_ORDER.index((kind, base))
    except ValueError:
        return len(_PRIORITY_ORDER)
    if _IS_ARM:
        ov_cpu = _PRIORITY_ORDER.index(("openvino", "CPU"))
        onnx_cpu = _PRIORITY_ORDER.index(("onnx", "cpu"))
        if (kind, base) == ("onnx", "cpu"):
            return ov_cpu        # take openvino-CPU's slot (higher priority)
        if (kind, base) == ("openvino", "CPU"):
            return onnx_cpu      # demote to onnx-cpu's old slot
    return rank


def priority_rank(slug: str) -> int:
    """Public alias of :func:`_priority_rank` — the canonical engine
    priority for a slug. Exposed so downstream consumers (notably
    ``depz_camera_viewer.model_selector``) can rank engines without
    duplicating :data:`_PRIORITY_ORDER` (and the ARM swap). Returns are
    sort keys (lower = higher priority); only relative order matters."""
    return _priority_rank(slug)


# Process-scoped cache for the OpenVINO probe. Host hardware doesn't
# change at runtime, but enumerate_engines() gets hit repeatedly —
# without caching we re-construct ov.Core(), re-query FULL_DEVICE_NAME
# for every device, and re-log the "hiding GPU.X" line each time.
# `[(dev_name, full_device_name)]` only; the live `ov.Core()` is
# instantiated on demand by callers that actually need to compile a
# model.
_OV_DEVICE_CACHE: Optional[list[tuple[str, str]]] = None

# Last engine list logged as a "runtimes" milestone — dedupes the STEP
# line across the many available_engines() calls per session.
_LAST_LOGGED_ENGINES: Optional[list[str]] = None


def _ov_device_list() -> list[tuple[str, str]]:
    """Cached list of ``(dev_name, full_device_name)`` for OpenVINO
    devices we expose. See :func:`_iter_ov_devices` for the filter rules
    and why the cache is safe (hardware doesn't change at runtime;
    callers can clear the cache via :func:`_reset_ov_cache` if probing
    after a hot-plug)."""
    global _OV_DEVICE_CACHE
    if _OV_DEVICE_CACHE is not None:
        return _OV_DEVICE_CACHE
    out: list[tuple[str, str]] = []
    try:
        import openvino as ov
    except Exception:
        # No OpenVINO in this env — a NORMAL state now that openvino is
        # an optional extra (ARM, macOS, and "pure" installs never have
        # it). DEBUG keeps it diagnosable without opening every session
        # log with a traceback.
        _LOGGER.debug("openvino not importable — no OpenVINO engines")
        _OV_DEVICE_CACHE = out
        return out
    core = ov.Core()
    for dev in core.available_devices:
        base = dev.split(".")[0].upper()
        try:
            full = str(core.get_property(dev, "FULL_DEVICE_NAME"))
        except Exception:
            _LOGGER.debug("FULL_DEVICE_NAME query failed for %s",
                          dev, exc_info=True)
            full = dev
        if base == "GPU" and "intel" not in full.lower():
            _LOGGER.info(
                "Hiding OpenVINO device %r (%s): OV's GPU plugin "
                "only compiles for Intel hardware; selecting it "
                "would silently fall back to CPU. Use onnx-cuda "
                "/ onnx-tensorrt for real NVIDIA acceleration "
                "(requires onnxruntime-gpu).",
                dev, full)
            continue
        out.append((dev, full))
    _OV_DEVICE_CACHE = out
    return out


def _reset_ov_cache() -> None:
    """Clear the cached OpenVINO device list. Call after a hardware
    change (rare — hot-plugged GPU, driver swap)."""
    global _OV_DEVICE_CACHE
    _OV_DEVICE_CACHE = None


def _iter_ov_devices() -> Iterator[tuple["object", str, str]]:
    """Yield ``(core, dev, full_name)`` for OpenVINO devices we expose.

    Filters out non-Intel GPUs. OpenVINO's GPU plugin enumerates
    **every OpenCL device** on the host — including NVIDIA cards
    when NVIDIA's OpenCL driver is installed (which is the default
    on any Windows box with a working CUDA setup). But the OV GPU
    plugin only knows how to compile kernels for **Intel** hardware,
    so picking a non-Intel ``GPU.N`` slug silently falls back to CPU
    at ``core.compile_model`` time. Surfacing those slugs in the UI
    is worse than not having them — the user clicks "NVIDIA" and
    inference runs on CPU with no error.

    Real NVIDIA acceleration lives under ``onnx-cuda`` /
    ``onnx-tensorrt`` (requires ``onnxruntime-gpu`` instead of the
    CPU-only ``onnxruntime`` wheel). AMD lives under ``onnx-rocm``.
    Intel iGPU/dGPU/NPU stay under OpenVINO."""
    cached = _ov_device_list()
    if not cached:
        return
    try:
        import openvino as ov
    except Exception:
        # Import worked during _ov_device_list but fails now — broken
        # install state worth knowing about.
        _LOGGER.warning("openvino import failed after successful probe",
                        exc_info=True)
        return
    core = ov.Core()
    for dev, full in cached:
        yield core, dev, full


def _is_hidden(slug: str) -> bool:
    """Whether `slug` is policy-hidden (see :data:`_HIDDEN_ENGINES`).
    OpenVINO multi-device suffixes (``GPU.0``) match on their base name,
    consistent with :func:`_priority_rank`."""
    kind, tail = parse_engine_slug(slug)
    return f"{kind}-{tail.split('.')[0]}" in _HIDDEN_ENGINES


def _probe_engines() -> list[str]:
    """Enumerate every loadable backend × device/EP on this host,
    unfiltered and unsorted. Non-Intel OV GPUs are already excluded at
    the probe level — see :func:`_iter_ov_devices` for why they can
    never work; that's capability, not policy, so no scope returns
    them."""
    out: list[str] = []

    for _core, dev, _full in _iter_ov_devices():
        out.append(f"openvino-{dev}")

    # ONNX Runtime EPs
    try:
        from .onnx_engine import list_available_eps
        for ep in list_available_eps():
            out.append(f"onnx-{ep}")
    except Exception:
        # onnxruntime absent or broken — the onnx-* family vanishes from
        # the engine dropdown; say why.
        _LOGGER.warning("ONNX Runtime EP probe failed — no onnx-* engines",
                        exc_info=True)
    return out


_SCOPES = ("available", "detected", "full", "hidden")


def list_engines(scope: str = "available") -> list[str]:
    """Engine slugs for the requested `scope`, **sorted by priority**.

    scope="available" (default)
        Loadable on this host right now, minus policy-hidden engines —
        what UIs should offer and recommenders should pick from.
        ``[0]`` is the best choice (== :func:`recommend_engine`).
    scope="detected"
        Loadable on this host *including* hidden engines — for
        diagnostics ("what did the probe actually see?").
    scope="full"
        Every (kind, device/EP) combination the SDK supports, straight
        from :data:`_PRIORITY_ORDER` — static, no host probing. Includes
        hidden entries; a slug's absence here means the SDK has no code
        path for it at all.
    scope="hidden"
        The policy-hidden slugs (see :data:`_HIDDEN_ENGINES`) — what
        separates "detected" from "available".
    """
    if scope not in _SCOPES:
        raise ValueError(
            f"unknown engine-list scope {scope!r}; expected one of {_SCOPES}")
    if scope == "full":
        return [f"{kind}-{tail}" for kind, tail in
                sorted(_PRIORITY_ORDER, key=lambda kt: _priority_rank(
                    f"{kt[0]}-{kt[1]}"))]
    if scope == "hidden":
        return sorted(_HIDDEN_ENGINES, key=_priority_rank)

    out = _probe_engines()
    if scope == "available":
        out = [s for s in out if not _is_hidden(s)]
    out.sort(key=_priority_rank)
    if scope == "available":
        # Milestone once per distinct result — the available list is hit
        # repeatedly (every state build), and the engine set only changes
        # on driver/install events.
        global _LAST_LOGGED_ENGINES
        if out != _LAST_LOGGED_ENGINES:
            _LAST_LOGGED_ENGINES = list(out)
            from .._log_util import log_step
            log_step(_LOGGER, "runtimes", engines=out)
    return out


def available_engines() -> list[str]:
    """Alias of ``list_engines("available")`` — the host's usable
    engines, priority-sorted. Kept as the short name for the
    overwhelmingly common case; see :func:`list_engines` for the other
    scopes."""
    return list_engines("available")


def recommend_engine() -> Optional[str]:
    """Best available engine slug for this host, or None if none loadable."""
    engines = available_engines()
    return engines[0] if engines else None


def make_engine(slug: str, model_path: str, **kwargs) -> InferenceEngine:
    """Construct the engine identified by `slug` against `model_path`.
    `slug` is one of the strings returned by :func:`available_engines`,
    e.g. ``"openvino-CPU"`` or ``"onnx-cuda"``. User input is accepted
    in any case — both kind and device/EP tail are normalized to the
    canonical form each backend expects (OpenVINO devices uppercase,
    ONNX EPs lowercase).

    Extra kwargs are passed through to the backend constructor.
    The most useful one is ``cache_dir=`` — backends that cache
    compiled models on disk (currently OpenVINO) will use
    ``<cache_dir>/<backend>/`` as the cache root. Defaults to
    :func:`depz_camera_sdk.get_cache_dir` when omitted.
    """
    kind, tail = parse_engine_slug(slug)
    if kind == "openvino":
        from .openvino_engine import OpenVINOEngine
        tail = tail.upper()
        engine: InferenceEngine = OpenVINOEngine(
            model_path, device=tail, **kwargs)
    elif kind == "onnx":
        from .onnx_engine import OnnxEngine
        # ONNX Runtime doesn't cache compiled models the same way; we
        # silently drop cache_dir to keep the registry signature uniform.
        tail = tail.lower()
        engine = OnnxEngine(model_path, ep=tail, **kwargs)
    else:
        raise ValueError(f"unknown engine kind in slug {slug!r}: {kind!r}")
    # Stamp the canonical slug on the engine so callers don't have to
    # carry it separately.
    engine.slug = f"{kind}-{tail}"
    return engine
