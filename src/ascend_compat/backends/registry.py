"""Backend protocol definition.

Every vendor backend implements this interface.  The core shim reads
these attributes to translate ``torch.cuda`` calls to the right API
without knowing anything about the specific vendor.

This is intentionally a simple class protocol (not an ABC) so that
backend modules can be defined with zero dependencies — they should
work even if the vendor's adapter library is not installed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class BackendInfo:
    """Protocol for a non-NVIDIA accelerator backend.

    Subclass this and fill in the class attributes.  The detection
    methods (``is_available``, ``device_count``) are called at runtime
    to probe hardware.  Everything else is static configuration.

    Example::

        class MyChipBackend(BackendInfo):
            name = "mychip"
            device_type = "mychip"
            adapter_module = "torch_mychip"
            collective_backend = "mccl"
            visible_devices_env = "MYCHIP_VISIBLE_DEVICES"

            @staticmethod
            def is_available() -> bool:
                try:
                    import torch_mychip
                    import torch
                    return torch.mychip.is_available()
                except Exception:
                    return False
    """

    # -- Static configuration (subclass MUST override) -----------------------

    #: Short identifier, e.g. ``"ascend"``, ``"cambricon"``
    name: str = ""

    #: PyTorch device type string, e.g. ``"npu"``, ``"mlu"``
    device_type: str = ""

    #: The pip-installable (or vendor-distributed) adapter package name
    adapter_module: str = ""

    #: Distributed collective library name, e.g. ``"hccl"``, ``"cncl"``
    collective_backend: str = ""

    #: Environment variable that replaces ``CUDA_VISIBLE_DEVICES``
    visible_devices_env: str = ""

    #: Human-readable display name
    display_name: str = ""

    #: Link to vendor documentation or adapter repo
    docs_url: str = ""

    # -- Runtime detection (subclass SHOULD override) ------------------------

    @staticmethod
    def is_available() -> bool:
        """Return True if this backend's hardware is detected and usable.

        This should import the adapter lazily and call its detection API.
        Must not raise — return False on any error.
        """
        return False

    @staticmethod
    def device_count() -> int:
        """Return the number of available devices (0 if unavailable)."""
        return 0

    @staticmethod
    def get_device_name(index: int = 0) -> str:
        """Return the device model name at the given index."""
        return "unknown"

    @classmethod
    def get_adapter_version(cls) -> Optional[str]:
        """Return the installed adapter version, or None if not installed."""
        try:
            import importlib
            mod = importlib.import_module(cls.adapter_module)
            return getattr(mod, "__version__", "unknown")
        except ImportError:
            return None

    @classmethod
    def summary(cls) -> Dict[str, Any]:
        """Return a dict summarizing this backend's status."""
        available = cls.is_available()
        return {
            "name": cls.name,
            "display_name": cls.display_name,
            "device_type": cls.device_type,
            "adapter_module": cls.adapter_module,
            "adapter_version": cls.get_adapter_version(),
            "available": available,
            "device_count": cls.device_count() if available else 0,
            "collective_backend": cls.collective_backend,
            "docs_url": cls.docs_url,
        }
