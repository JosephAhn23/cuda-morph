"""Memory management API — backend-agnostic wrappers.

Replaces ``torch.cuda.memory_allocated()``, ``torch.cuda.empty_cache()``,
etc. with calls that route to the detected backend.

Usage::

    import ascend_compat.memory as memory

    print(f"Allocated: {memory.memory_allocated() / 1e9:.2f} GB")
    memory.empty_cache()
"""

from __future__ import annotations

from typing import Any, Optional

from ascend_compat._logging import get_logger

logger = get_logger(__name__)


def _get_backend_module() -> Any:
    """Return the torch device module for the preferred backend."""
    from ascend_compat.device import _get_backend_module
    return _get_backend_module()


def memory_allocated(device: Optional[int] = None) -> int:
    """Return current memory usage in bytes on the given device."""
    mod = _get_backend_module()
    if mod is None or not hasattr(mod, "memory_allocated"):
        return 0
    return mod.memory_allocated(device) if device is not None else mod.memory_allocated()


def max_memory_allocated(device: Optional[int] = None) -> int:
    """Return peak memory usage in bytes on the given device."""
    mod = _get_backend_module()
    if mod is None or not hasattr(mod, "max_memory_allocated"):
        return 0
    return mod.max_memory_allocated(device) if device is not None else mod.max_memory_allocated()


def memory_reserved(device: Optional[int] = None) -> int:
    """Return current reserved (cached) memory in bytes."""
    mod = _get_backend_module()
    if mod is None or not hasattr(mod, "memory_reserved"):
        return 0
    return mod.memory_reserved(device) if device is not None else mod.memory_reserved()


def max_memory_reserved(device: Optional[int] = None) -> int:
    """Return peak reserved (cached) memory in bytes."""
    mod = _get_backend_module()
    if mod is None or not hasattr(mod, "max_memory_reserved"):
        return 0
    return mod.max_memory_reserved(device) if device is not None else mod.max_memory_reserved()


def empty_cache() -> None:
    """Release all unoccupied cached memory."""
    mod = _get_backend_module()
    if mod is not None and hasattr(mod, "empty_cache"):
        mod.empty_cache()


def reset_peak_memory_stats(device: Optional[int] = None) -> None:
    """Reset peak memory tracking statistics."""
    mod = _get_backend_module()
    if mod is not None and hasattr(mod, "reset_peak_memory_stats"):
        if device is not None:
            mod.reset_peak_memory_stats(device)
        else:
            mod.reset_peak_memory_stats()


def memory_summary(device: Optional[int] = None) -> str:
    """Return a human-readable memory summary string."""
    mod = _get_backend_module()
    if mod is None or not hasattr(mod, "memory_summary"):
        return "No accelerator available — no memory to report"
    return mod.memory_summary(device) if device is not None else mod.memory_summary()
