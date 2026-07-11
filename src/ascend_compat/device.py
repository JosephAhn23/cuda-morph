"""Device management API — backend-agnostic wrappers over torch.cuda/torch.npu/etc.

These are the **real** functions that ``cuda-morph check`` and ``cuda-morph port``
reference in their suggestions.  Each function delegates to whatever backend is
currently detected (Ascend NPU, AMD ROCm, Intel XPU, or CPU fallback).

Usage::

    import ascend_compat.device as device

    if device.is_available():
        device.set_device(0)
        name = device.get_device_name()
"""

from __future__ import annotations

from typing import Any

from ascend_compat._backend import Backend, get_torch, preferred_backend
from ascend_compat._logging import get_logger

logger = get_logger(__name__)


def _get_backend_module() -> Any:
    """Return the torch device module for the preferred backend (torch.npu, torch.cuda, etc.)."""
    torch = get_torch()
    backend = preferred_backend()
    _MODULE_MAP = {
        Backend.NPU: lambda: getattr(torch, "npu", None),
        Backend.MLU: lambda: getattr(torch, "mlu", None),
        Backend.XPU: lambda: getattr(torch, "xpu", None),
        Backend.ROCM: lambda: torch.cuda,
        Backend.CUDA: lambda: torch.cuda,
    }
    getter = _MODULE_MAP.get(backend)
    return getter() if getter else None


def is_available() -> bool:
    """Return True if any accelerator backend is available."""
    mod = _get_backend_module()
    if mod is None:
        return False
    return mod.is_available()


def device_count() -> int:
    """Return the number of available accelerator devices."""
    mod = _get_backend_module()
    if mod is None:
        return 0
    return mod.device_count()


def current_device() -> int:
    """Return the index of the currently selected device."""
    mod = _get_backend_module()
    if mod is None:
        return 0
    return mod.current_device()


def set_device(device: int) -> None:
    """Set the current device by index."""
    mod = _get_backend_module()
    if mod is not None:
        mod.set_device(device)


def get_device_name(device: int = 0) -> str:
    """Return the name of the device at the given index."""
    mod = _get_backend_module()
    if mod is None:
        return "cpu"
    return mod.get_device_name(device)


def get_device_properties(device: int = 0) -> Any:
    """Return device properties for the given index."""
    mod = _get_backend_module()
    if mod is None:
        return None
    if hasattr(mod, "get_device_properties"):
        return mod.get_device_properties(device)
    return None


def get_device_string() -> str:
    """Return the correct device type string for the preferred backend.

    Returns ``"npu"``, ``"cuda"``, ``"xpu"``, ``"mlu"``, or ``"cpu"``.
    """
    from ascend_compat._backend import translate_device_string
    return translate_device_string("cuda")


def to_device(obj: Any, device: Any = None) -> Any:
    """Move a tensor or module to the preferred accelerator.

    This replaces ``.cuda()`` calls with backend-aware placement.

    Args:
        obj: A ``torch.Tensor`` or ``torch.nn.Module``.
        device: Optional device index or string.  If None, uses the
            preferred backend's default device.

    Returns:
        The object moved to the target device.
    """
    torch = get_torch()
    backend = preferred_backend()

    if device is None:
        device = get_device_string()

    if isinstance(device, str) and device.startswith("cuda"):
        from ascend_compat._backend import translate_device_string
        device = translate_device_string(device)

    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, torch.nn.Module):
        return obj.to(device)
    else:
        raise TypeError(f"Cannot move {type(obj).__name__} to device")
