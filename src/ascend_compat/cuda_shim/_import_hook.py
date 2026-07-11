"""sys.meta_path import hook that intercepts ``import torch.cuda``.

Why an import hook?
-------------------
Monkey-patching ``torch.cuda`` after import only catches code that accesses
attributes *after* our patch runs.  But many libraries do
``from torch.cuda.amp import autocast`` at module load time — before
``import ascend_compat`` gets a chance to patch.

By installing a ``sys.meta_path`` finder, we intercept the ``import torch.cuda``
statement *before* Python resolves it, and can inject our shim module or ensure
torch_npu is imported first (which does its own PrivateUse1 registration).

Implementation
--------------
We DON'T replace torch.cuda with a fake module.  That's too fragile — torch
internals access torch.cuda at the C level and would break.  Instead we:

1. Ensure torch_npu is imported first (triggering its PrivateUse1 registration)
2. Let the real torch.cuda import proceed
3. Post-import, apply our monkey-patches

This two-phase approach is more robust than either approach alone.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any, Optional, Sequence

from ascend_compat._logging import get_logger

logger = get_logger(__name__)

_hook_installed = False


class _AscendCompatFinder:
    """Meta path finder that ensures torch_npu is loaded before torch.cuda.

    This does NOT block or replace torch.cuda — it just makes sure the
    NPU backend is registered before any CUDA code runs.
    """

    def __init__(self) -> None:
        self._processing = False  # re-entrancy guard

    def find_module(
        self, fullname: str, path: Optional[Sequence[str]] = None
    ) -> Optional["_AscendCompatFinder"]:
        """Called by Python's import system for every import statement.

        We only intercept ``torch.cuda`` and its submodules.
        """
        if self._processing:
            return None  # Prevent infinite recursion

        if fullname == "torch.cuda" or fullname.startswith("torch.cuda."):
            return self
        return None

    def load_module(self, fullname: str) -> Any:
        """Called when find_module returns self.

        We ensure torch_npu is imported, then let the real import proceed.
        """
        self._processing = True
        try:
            # Ensure torch_npu is imported first (registers PrivateUse1 backend)
            _ensure_torch_npu()

            # Now let the real import proceed
            if fullname in sys.modules:
                return sys.modules[fullname]

            # Temporarily remove ourselves to avoid recursion
            return importlib.import_module(fullname)
        finally:
            self._processing = False


def _ensure_torch_npu() -> None:
    """Import torch_npu if available, triggering PrivateUse1 registration.

    Since PyTorch 2.5, torch may auto-load torch_npu via accelerator
    autoloading.  We call this explicitly for older versions.
    """
    if "torch_npu" in sys.modules:
        return  # Already loaded

    try:
        import torch_npu  # type: ignore[import-untyped]  # noqa: F401
        logger.debug("torch_npu imported via import hook — PrivateUse1 backend registered")
    except ImportError:
        logger.debug("torch_npu not available — import hook is a no-op")


def install_import_hook() -> None:
    """Install the meta path finder.

    Idempotent — safe to call multiple times.
    """
    global _hook_installed  # noqa: PLW0603
    if _hook_installed:
        return

    # Insert at the beginning so we run before other finders
    finder = _AscendCompatFinder()
    sys.meta_path.insert(0, finder)
    _hook_installed = True
    logger.debug("Import hook installed on sys.meta_path")


def uninstall_import_hook() -> None:
    """Remove the meta path finder.

    Primarily for testing.
    """
    global _hook_installed  # noqa: PLW0603
    sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, _AscendCompatFinder)]
    _hook_installed = False
    logger.debug("Import hook removed from sys.meta_path")
