"""Stream and synchronization API — backend-agnostic wrappers.

Replaces ``torch.cuda.synchronize()``, ``torch.cuda.Stream()``, etc.
with calls that route to the detected backend.

Usage::

    import ascend_compat.streams as streams

    streams.synchronize()
    with streams.Stream() as s:
        # ops run on this stream
        pass
"""

from __future__ import annotations

from typing import Any, Optional

from ascend_compat._logging import get_logger

logger = get_logger(__name__)


def _get_backend_module() -> Any:
    """Return the torch device module for the preferred backend."""
    from ascend_compat.device import _get_backend_module
    return _get_backend_module()


def synchronize(device: Optional[int] = None) -> None:
    """Wait for all operations on the current device to complete."""
    mod = _get_backend_module()
    if mod is not None and hasattr(mod, "synchronize"):
        if device is not None:
            mod.synchronize(device)
        else:
            mod.synchronize()


def Stream(*args: Any, **kwargs: Any) -> Any:
    """Create a device stream (routes to the backend's Stream class)."""
    mod = _get_backend_module()
    if mod is not None and hasattr(mod, "Stream"):
        return mod.Stream(*args, **kwargs)
    raise RuntimeError("No accelerator backend available for stream creation")


def Event(*args: Any, **kwargs: Any) -> Any:
    """Create a device event (routes to the backend's Event class)."""
    mod = _get_backend_module()
    if mod is not None and hasattr(mod, "Event"):
        return mod.Event(*args, **kwargs)
    raise RuntimeError("No accelerator backend available for event creation")


def current_stream(device: Optional[int] = None) -> Any:
    """Return the currently active stream for the given device."""
    mod = _get_backend_module()
    if mod is not None and hasattr(mod, "current_stream"):
        return mod.current_stream(device) if device is not None else mod.current_stream()
    raise RuntimeError("No accelerator backend available")
