"""Backend detection and capability probing.

This module is the single source of truth for "what hardware is available right
now?"  Every other module in cuda-morph imports from here rather than
re-running its own detection logic.

Architecture note
-----------------
We intentionally *lazy-import* ``torch``, ``torch_npu``, etc. so that
cuda-morph can be imported even when PyTorch isn't installed (useful
for the CLI static-analysis tool ``cuda-morph check``).

Multi-backend support
---------------------
cuda-morph supports multiple domestic AI chip backends:

1. **Ascend NPU** via ``torch_npu`` (Huawei)
2. **Cambricon MLU** via ``torch_mlu`` (Cambricon)
3. **NVIDIA CUDA** via ``torch.cuda`` (reference/fallback)
4. **CPU** — always available, used for development & CI

Backend detection uses the pluggable registry in ``backends/``.  Each
backend module implements a common protocol (``BackendInfo``).  The detection
loop probes each registered backend in priority order and selects the first
one that reports hardware available.

Why a dedicated module?
-----------------------
Centralising detection avoids import-order bugs.  For example, if ``device.py``
and ``memory.py`` both independently tried ``import torch_npu``, a race or
circular-import could surface.  By funnelling everything through ``_backend``
we guarantee a single, well-ordered detection pass.
"""

from __future__ import annotations

import enum
import functools
from typing import Any, Dict, Optional, Type

from ascend_compat._logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Backend enumeration
# ---------------------------------------------------------------------------


class Backend(enum.Enum):
    """Available compute backends, ordered by preference."""

    NPU = "npu"      # Huawei Ascend via torch_npu
    MLU = "mlu"      # Cambricon via torch_mlu
    ROCM = "rocm"    # AMD via ROCm/HIP (presents as "cuda" device)
    XPU = "xpu"      # Intel via IPEX/Level Zero
    CUDA = "cuda"    # NVIDIA via torch.cuda
    CPU = "cpu"      # Always available


# Map backend device type strings to enum values
_BACKEND_DEVICE_TYPES: Dict[str, Backend] = {
    "npu": Backend.NPU,
    "mlu": Backend.MLU,
    "rocm": Backend.ROCM,
    "xpu": Backend.XPU,
    "cuda": Backend.CUDA,
    "cpu": Backend.CPU,
}


# ---------------------------------------------------------------------------
# Lazy module references (populated on first access)
# ---------------------------------------------------------------------------

_torch: Optional[Any] = None
_torch_npu: Optional[Any] = None


def _import_torch() -> Any:
    """Lazily import torch, caching the result."""
    global _torch  # noqa: PLW0603
    if _torch is None:
        try:
            import torch  # type: ignore[import-untyped]
            _torch = torch
        except ImportError:
            raise ImportError(
                "PyTorch is required but not installed. "
                "Install it with: pip install torch>=2.0"
            ) from None
    return _torch


def _import_torch_npu() -> Optional[Any]:
    """Lazily import torch_npu, returning None if unavailable.

    torch_npu is Huawei's official PyTorch adapter for Ascend NPUs.
    It monkey-patches torch to add NPU device support.  If it isn't
    installed, we gracefully fall back to CUDA or CPU.

    See: https://gitee.com/ascend/pytorch
    """
    global _torch_npu  # noqa: PLW0603
    if _torch_npu is None:
        try:
            import torch_npu  # type: ignore[import-untyped]
            _torch_npu = torch_npu
            logger.debug("torch_npu imported successfully — Ascend backend available")
        except ImportError:
            logger.debug("torch_npu not found — Ascend backend unavailable")
            _torch_npu = False  # sentinel: tried and failed
    return _torch_npu if _torch_npu is not False else None


# ---------------------------------------------------------------------------
# Active backend tracking
# ---------------------------------------------------------------------------

_active_backend_info: Optional[Any] = None  # BackendInfo subclass, set by activate()


def get_active_backend_info() -> Optional[Any]:
    """Return the active backend's BackendInfo, or None if not set."""
    return _active_backend_info


def set_active_backend_info(info: Optional[Any]) -> None:
    """Set the active backend info (called by activate())."""
    global _active_backend_info  # noqa: PLW0603
    _active_backend_info = info


# ---------------------------------------------------------------------------
# Detection logic
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def detect_backends() -> tuple[Backend, ...]:
    """Probe the system and return all available backends, best-first.

    The result is cached for the lifetime of the process because hardware
    doesn't change at runtime.

    Detection order:
    1. Check each registered backend in the pluggable registry
    2. Check NVIDIA CUDA
    3. CPU (always available)

    Returns:
        Tuple of :class:`Backend` values, ordered from most-preferred to
        least-preferred.
    """
    available: list[Backend] = []

    # 1. Check pluggable backends from the registry
    try:
        from ascend_compat.backends import BACKEND_REGISTRY
        for name, backend_cls in BACKEND_REGISTRY.items():
            try:
                if backend_cls.is_available():
                    device_type = backend_cls.device_type
                    backend_enum = _BACKEND_DEVICE_TYPES.get(device_type)
                    if backend_enum and backend_enum not in available:
                        available.append(backend_enum)
                        logger.info(
                            "%s detected (%d device(s))",
                            backend_cls.display_name,
                            backend_cls.device_count(),
                        )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Backend '%s' detection failed: %s", name, exc
                )
    except ImportError:
        # Fallback: probe directly if backends package fails to import
        logger.debug("backends package not available, using legacy detection")
        _detect_legacy(available)

    # 2. Check for NVIDIA CUDA (if not already found via registry)
    if Backend.CUDA not in available:
        torch = _import_torch()
        if torch.cuda.is_available():
            available.append(Backend.CUDA)
            logger.info(
                "NVIDIA CUDA detected (%d device(s))",
                torch.cuda.device_count(),
            )

    # 3. CPU is always available
    if Backend.CPU not in available:
        available.append(Backend.CPU)

    logger.debug("Detected backends (preference order): %s", available)
    return tuple(available)


def _detect_legacy(available: list[Backend]) -> None:
    """Legacy detection path (before pluggable backends existed).

    This is the fallback for when the ``backends`` subpackage can't be
    imported (e.g. during early development or if the package structure
    changes).
    """
    # Check for Ascend NPU
    npu_mod = _import_torch_npu()
    if npu_mod is not None:
        torch = _import_torch()
        try:
            if hasattr(torch, "npu") and torch.npu.is_available():
                available.append(Backend.NPU)
                logger.info(
                    "Ascend NPU detected (%d device(s))",
                    torch.npu.device_count(),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("torch_npu installed but NPU detection failed: %s", exc)


@functools.lru_cache(maxsize=1)
def preferred_backend() -> Backend:
    """Return the single best backend for this system.

    This drives the default behaviour of ``cuda-morph`` — all CUDA
    calls are routed to whichever backend this function returns.
    """
    return detect_backends()[0]


# ---------------------------------------------------------------------------
# Convenience predicates
# ---------------------------------------------------------------------------


def has_npu() -> bool:
    """Return True if at least one Ascend NPU is usable."""
    return Backend.NPU in detect_backends()


def has_mlu() -> bool:
    """Return True if at least one Cambricon MLU is usable."""
    return Backend.MLU in detect_backends()


def has_rocm() -> bool:
    """Return True if AMD ROCm GPU is detected."""
    return Backend.ROCM in detect_backends()


def has_xpu() -> bool:
    """Return True if at least one Intel XPU is usable."""
    return Backend.XPU in detect_backends()


def has_cuda() -> bool:
    """Return True if at least one NVIDIA GPU is usable."""
    return Backend.CUDA in detect_backends()


def get_torch() -> Any:
    """Return the ``torch`` module (importing it if necessary).

    This is the canonical way for other cuda-morph modules to get a
    reference to torch without redundant try/except blocks.
    """
    return _import_torch()


def get_torch_npu() -> Optional[Any]:
    """Return the ``torch_npu`` module, or None if not installed."""
    return _import_torch_npu()


# ---------------------------------------------------------------------------
# Device-string translation
# ---------------------------------------------------------------------------


def translate_device_string(device: str) -> str:
    """Translate a CUDA device string to the appropriate backend string.

    Mapping rules:
    - If a domestic backend (NPU, MLU) is preferred, ``"cuda"`` → backend device type
    - If CUDA is preferred (or we're on CPU), return the string unchanged.

    Args:
        device: A PyTorch device string, e.g. ``"cuda"``, ``"cuda:0"``,
            ``"cpu"``, ``"npu:1"``.

    Returns:
        The translated device string.

    Examples::

        # On an Ascend system:
        translate_device_string("cuda")    # → "npu"
        translate_device_string("cuda:2")  # → "npu:2"

        # On a Cambricon system:
        translate_device_string("cuda")    # → "mlu"
        translate_device_string("cuda:0")  # → "mlu:0"

        # On CPU:
        translate_device_string("cuda")    # → "cpu"
    """
    backend = preferred_backend()

    # Backends that need "cuda" → their device type translation
    _TRANSLATE_BACKENDS = {
        Backend.NPU: "npu",
        Backend.MLU: "mlu",
        Backend.XPU: "xpu",
        # ROCm does NOT need translation — it presents as "cuda" via HIP
    }

    if backend in _TRANSLATE_BACKENDS and device.startswith("cuda"):
        target_type = _TRANSLATE_BACKENDS[backend]
        translated = device.replace("cuda", target_type, 1)
        logger.debug("Device string translated: %r → %r", device, translated)
        return translated

    if backend == Backend.CPU and device.startswith("cuda"):
        # No accelerator at all — fall back to CPU so the code doesn't crash.
        translated = "cpu"
        logger.warning(
            "No GPU/NPU/MLU available — translating device %r → 'cpu'. "
            "Performance will be significantly lower.",
            device,
        )
        return translated

    return device
