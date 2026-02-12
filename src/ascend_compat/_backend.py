"""Backend detection and capability probing.

This module is the single source of truth for "what hardware is available right
now?"  Every other module in ascend-compat imports from here rather than
re-running its own detection logic.

Architecture note
-----------------
We intentionally *lazy-import* ``torch``, ``torch_npu``, etc. so that
``ascend-compat`` can be imported even when PyTorch isn't installed (useful
for the CLI static-analysis tool ``ascend-compat check``).

Detection priority
------------------
1. **Ascend NPU** via ``torch_npu``  – preferred when available
2. **NVIDIA CUDA** via ``torch.cuda`` – used as reference / fallback
3. **CPU** – always available, used for development & CI

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
from typing import Any, Optional

from ascend_compat._logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Backend enumeration
# ---------------------------------------------------------------------------


class Backend(enum.Enum):
    """Available compute backends, ordered by preference."""

    NPU = "npu"      # Huawei Ascend via torch_npu
    CUDA = "cuda"    # NVIDIA via torch.cuda
    CPU = "cpu"      # Always available


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
# Detection logic
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def detect_backends() -> tuple[Backend, ...]:
    """Probe the system and return all available backends, best-first.

    The result is cached for the lifetime of the process because hardware
    doesn't change at runtime.

    Returns:
        Tuple of :class:`Backend` values, ordered from most-preferred to
        least-preferred.
    """
    available: list[Backend] = []

    # 1. Check for Ascend NPU via torch_npu
    npu_mod = _import_torch_npu()
    if npu_mod is not None:
        torch = _import_torch()
        try:
            # torch_npu patches torch to add torch.npu.is_available()
            if hasattr(torch, "npu") and torch.npu.is_available():
                available.append(Backend.NPU)
                logger.info(
                    "Ascend NPU detected (%d device(s))",
                    torch.npu.device_count(),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("torch_npu is installed but NPU detection failed: %s", exc)

    # 2. Check for NVIDIA CUDA
    torch = _import_torch()
    if torch.cuda.is_available():
        available.append(Backend.CUDA)
        logger.info(
            "NVIDIA CUDA detected (%d device(s))",
            torch.cuda.device_count(),
        )

    # 3. CPU is always available
    available.append(Backend.CPU)

    logger.debug("Detected backends (preference order): %s", available)
    return tuple(available)


@functools.lru_cache(maxsize=1)
def preferred_backend() -> Backend:
    """Return the single best backend for this system.

    This drives the default behaviour of `import ascend_compat` — all CUDA
    calls are routed to whichever backend this function returns.
    """
    return detect_backends()[0]


# ---------------------------------------------------------------------------
# Convenience predicates
# ---------------------------------------------------------------------------


def has_npu() -> bool:
    """Return True if at least one Ascend NPU is usable."""
    return Backend.NPU in detect_backends()


def has_cuda() -> bool:
    """Return True if at least one NVIDIA GPU is usable."""
    return Backend.CUDA in detect_backends()


def get_torch() -> Any:
    """Return the ``torch`` module (importing it if necessary).

    This is the canonical way for other ascend-compat modules to get a
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
    - If NPU is the preferred backend, ``"cuda"`` → ``"npu"``
      and ``"cuda:N"`` → ``"npu:N"``
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
        translate_device_string("cpu")     # → "cpu"  (unchanged)
    """
    backend = preferred_backend()

    if backend == Backend.NPU and device.startswith("cuda"):
        translated = device.replace("cuda", "npu", 1)
        logger.debug("Device string translated: %r → %r", device, translated)
        return translated

    if backend == Backend.CPU and device.startswith("cuda"):
        # No accelerator at all — fall back to CPU so the code doesn't crash.
        translated = "cpu"
        logger.warning(
            "No GPU/NPU available — translating device %r → 'cpu'. "
            "Performance will be significantly lower.",
            device,
        )
        return translated

    return device
