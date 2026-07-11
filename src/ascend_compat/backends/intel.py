"""Intel GPU backend (Intel Extension for PyTorch / Level Zero).

Intel's data center GPUs (Flex, Max series) and integrated GPUs (Arc)
are supported through Intel Extension for PyTorch (IPEX), which registers
the ``"xpu"`` device type via PyTorch's PrivateUse1 mechanism.

Hardware: Intel Data Center GPU Max 1550, Flex 170; Arc A770
Runtime: oneAPI (Level Zero, oneMKL, oneDNN)
Collective: oneCCL (Intel oneAPI Collective Communications Library)
Adapter: intel-extension-for-pytorch (https://github.com/intel/intel-extension-for-pytorch)

STATUS: BACKEND STUB
--------------------
Detection logic is implemented.  Ecosystem patches are NOT yet implemented.
"""

from __future__ import annotations

from typing import Optional

from ascend_compat.backends.registry import BackendInfo


class IntelBackend(BackendInfo):
    """Intel GPU via Intel Extension for PyTorch (IPEX)."""

    name = "intel"
    device_type = "xpu"
    adapter_module = "intel_extension_for_pytorch"
    collective_backend = "ccl"
    visible_devices_env = "ZE_AFFINITY_MASK"
    display_name = "Intel GPU (XPU)"
    docs_url = "https://github.com/intel/intel-extension-for-pytorch"

    @staticmethod
    def is_available() -> bool:
        """Check if Intel GPU hardware is present and IPEX is loaded."""
        try:
            import intel_extension_for_pytorch  # type: ignore[import-untyped]  # noqa: F401
            import torch
            return hasattr(torch, "xpu") and torch.xpu.is_available()
        except Exception:
            return False

    @staticmethod
    def device_count() -> int:
        """Return number of Intel GPU devices."""
        try:
            import torch
            if hasattr(torch, "xpu"):
                return torch.xpu.device_count()
        except Exception:
            pass
        return 0

    @staticmethod
    def get_device_name(index: int = 0) -> str:
        """Return the Intel GPU model name."""
        try:
            import torch
            if hasattr(torch, "xpu"):
                return torch.xpu.get_device_name(index)
        except Exception:
            pass
        return "Intel GPU (unknown model)"

    @classmethod
    def get_adapter_version(cls) -> Optional[str]:
        """Return the IPEX version."""
        try:
            import intel_extension_for_pytorch as ipex  # type: ignore[import-untyped]
            return getattr(ipex, "__version__", "unknown")
        except ImportError:
            return None
