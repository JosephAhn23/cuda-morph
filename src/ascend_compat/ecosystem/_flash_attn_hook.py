"""Import hook that registers ascend-compat's flash_attn shim as a package.

When activated, ``from flash_attn import flash_attn_func`` will resolve to
our compatibility wrapper around ``torch_npu.npu_fusion_attention`` — with
zero code changes required in the user's script.

Implementation uses the modern importlib Finder/Loader API (PEP 302 / 451)
with ``find_module`` returning a loader that provides ``create_module`` and
``exec_module``.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
from types import ModuleType
from typing import Any, Optional, Sequence

from ascend_compat._logging import get_logger

logger = get_logger(__name__)

_hook_installed = False


class _FlashAttnLoader(importlib.abc.Loader):
    """Loader that returns our flash_attn shim module."""

    def create_module(self, spec: Any) -> Optional[ModuleType]:
        """Return None to let Python create a default module object."""
        return None

    def exec_module(self, module: ModuleType) -> None:
        """Populate the module with our shim's attributes."""
        from ascend_compat.ecosystem import flash_attn as fa_shim

        # Copy all public attributes from our shim
        for attr in dir(fa_shim):
            if not attr.startswith("_"):
                setattr(module, attr, getattr(fa_shim, attr))

        # Also set essential dunder attributes
        module.__path__ = []  # Make it look like a package
        module.__package__ = module.__name__
        module.__loader__ = self
        logger.debug("%s → ascend_compat.ecosystem.flash_attn (shim)", module.__name__)


class _FlashAttnFinder(importlib.abc.MetaPathFinder):
    """Meta path finder that intercepts ``import flash_attn``."""

    def __init__(self) -> None:
        self._loader = _FlashAttnLoader()

    def find_module(
        self, fullname: str, path: Optional[Sequence[str]] = None
    ) -> Optional[importlib.abc.Loader]:
        """Legacy find_module for older Python compat."""
        spec = self.find_spec(fullname, path)
        if spec is not None:
            return spec.loader
        return None

    def find_spec(
        self,
        fullname: str,
        path: Optional[Sequence[str]],
        target: Optional[ModuleType] = None,
    ) -> Optional[importlib.machinery.ModuleSpec]:
        """Called by Python's import system to find a module."""
        # Only intercept flash_attn and submodules
        if fullname != "flash_attn" and not fullname.startswith("flash_attn."):
            return None

        # If already in sys.modules, let Python handle it
        if fullname in sys.modules:
            return None

        # Try to see if the real flash_attn is installable by checking if
        # any other finder can handle it (skip ourselves)
        for finder in sys.meta_path:
            if finder is self:
                continue
            find_spec = getattr(finder, "find_spec", None)
            if find_spec is not None:
                try:
                    spec = find_spec(fullname, path, target)
                    if spec is not None:
                        # Real package found — don't override
                        logger.debug("Real flash_attn package found — not overriding")
                        return None
                except (ImportError, AttributeError, ValueError):
                    continue

        # No real package — serve our shim
        return importlib.machinery.ModuleSpec(
            fullname,
            self._loader,
            is_package=(fullname == "flash_attn"),
        )


def install_flash_attn_hook() -> None:
    """Install the flash_attn import hook.  Idempotent."""
    global _hook_installed  # noqa: PLW0603
    if _hook_installed:
        return

    finder = _FlashAttnFinder()
    sys.meta_path.append(finder)  # Append — lower priority than real packages
    _hook_installed = True
    logger.debug("flash_attn import hook installed")


def uninstall_flash_attn_hook() -> None:
    """Remove the flash_attn import hook."""
    global _hook_installed  # noqa: PLW0603
    sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, _FlashAttnFinder)]
    _hook_installed = False

    # Clean up sys.modules entries we created
    from ascend_compat.ecosystem import flash_attn as fa_shim

    for key in list(sys.modules):
        if key == "flash_attn" or key.startswith("flash_attn."):
            mod = sys.modules.get(key)
            if mod is not None:
                loader = getattr(mod, "__loader__", None)
                if isinstance(loader, _FlashAttnLoader):
                    del sys.modules[key]

    logger.debug("flash_attn import hook removed")
