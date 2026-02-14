"""Operation wrappers — backend-agnostic AMP, seeding, and graph utilities.

Replaces ``torch.cuda.amp.autocast``, ``torch.cuda.manual_seed()``,
``torch.cuda.CUDAGraph``, etc. with calls that route to the detected backend.

Usage::

    import ascend_compat.ops as ops

    with ops.autocast():
        output = model(input)

    ops.manual_seed(42)
"""

from __future__ import annotations

from typing import Any

from ascend_compat._backend import Backend, get_torch, preferred_backend
from ascend_compat._logging import get_logger

logger = get_logger(__name__)


def _get_device_type() -> str:
    """Return the device type string for the preferred backend."""
    backend = preferred_backend()
    _TYPE_MAP = {
        Backend.NPU: "npu",
        Backend.MLU: "mlu",
        Backend.XPU: "xpu",
        Backend.ROCM: "cuda",
        Backend.CUDA: "cuda",
        Backend.CPU: "cpu",
    }
    return _TYPE_MAP.get(backend, "cpu")


def autocast(*args: Any, **kwargs: Any) -> Any:
    """Backend-aware autocast context manager.

    Wraps ``torch.amp.autocast`` with the correct device_type for the
    current backend.  If ``device_type`` is passed as ``"cuda"``, it is
    transparently remapped to the detected backend's type.
    """
    torch = get_torch()
    device_type = kwargs.pop("device_type", None)
    if device_type is None and args:
        device_type = args[0]
        args = args[1:]
    if device_type is None or device_type == "cuda":
        device_type = _get_device_type()
    return torch.amp.autocast(device_type, *args, **kwargs)


def GradScaler(*args: Any, **kwargs: Any) -> Any:
    """Backend-aware GradScaler.

    Wraps ``torch.amp.GradScaler`` with the correct device for the
    current backend.
    """
    torch = get_torch()
    device = kwargs.pop("device", None)
    if device is None or device == "cuda":
        device = _get_device_type()
    kwargs["device"] = device
    return torch.amp.GradScaler(*args, **kwargs)


def manual_seed(seed: int) -> None:
    """Set the random seed on the current accelerator device."""
    torch = get_torch()
    backend = preferred_backend()
    mod_map = {
        Backend.NPU: lambda: getattr(torch, "npu", None),
        Backend.MLU: lambda: getattr(torch, "mlu", None),
        Backend.XPU: lambda: getattr(torch, "xpu", None),
        Backend.ROCM: lambda: torch.cuda,
        Backend.CUDA: lambda: torch.cuda,
    }
    getter = mod_map.get(backend)
    mod = getter() if getter else None
    if mod is not None and hasattr(mod, "manual_seed"):
        mod.manual_seed(seed)
    # Always seed CPU too
    torch.manual_seed(seed)


def manual_seed_all(seed: int) -> None:
    """Set the random seed on all accelerator devices."""
    torch = get_torch()
    backend = preferred_backend()
    mod_map = {
        Backend.NPU: lambda: getattr(torch, "npu", None),
        Backend.MLU: lambda: getattr(torch, "mlu", None),
        Backend.XPU: lambda: getattr(torch, "xpu", None),
        Backend.ROCM: lambda: torch.cuda,
        Backend.CUDA: lambda: torch.cuda,
    }
    getter = mod_map.get(backend)
    mod = getter() if getter else None
    if mod is not None and hasattr(mod, "manual_seed_all"):
        mod.manual_seed_all(seed)
    torch.manual_seed(seed)


def get_distributed_backend() -> str:
    """Return the correct distributed backend name for the current hardware.

    Returns ``"hccl"`` for Ascend, ``"cncl"`` for Cambricon, ``"rccl"`` for
    AMD ROCm, ``"ccl"`` for Intel, ``"nccl"`` for NVIDIA, or ``"gloo"`` for CPU.
    """
    backend = preferred_backend()
    _DIST_MAP = {
        Backend.NPU: "hccl",
        Backend.MLU: "cncl",
        Backend.ROCM: "rccl",
        Backend.XPU: "ccl",
        Backend.CUDA: "nccl",
        Backend.CPU: "gloo",
    }
    return _DIST_MAP.get(backend, "gloo")


def graph_mode() -> Any:
    """Context manager for graph capture (where supported).

    On backends that support graph capture (CUDA Graphs, NPU graph mode),
    returns the appropriate context manager.  On unsupported backends,
    returns a no-op context manager.

    Note: CUDA Graphs are NOT supported on most non-NVIDIA backends.
    This wrapper provides a graceful fallback.
    """
    import contextlib
    backend = preferred_backend()

    if backend == Backend.CUDA:
        torch = get_torch()
        if hasattr(torch.cuda, "graph"):
            return torch.cuda.graph()

    logger.warning(
        "Graph capture is not supported on %s — running eagerly", backend.value
    )
    return contextlib.nullcontext()
