"""Intel GPU backend (native torch.xpu, with legacy IPEX fallback).

Intel's data center GPUs (Flex, Max series) and integrated GPUs (Arc)
register the ``"xpu"`` device type via PyTorch's PrivateUse1 mechanism.

As of PyTorch 2.9, Intel folded the standalone Intel Extension for
PyTorch (IPEX) into core ``torch`` — ``torch.xpu`` works without
importing anything extra, and IPEX itself is no longer actively
released (see https://github.com/intel/intel-extension-for-pytorch).
On PyTorch <2.9, ``torch.xpu`` still requires ``intel_extension_for_pytorch``
to be imported first to register the device. This backend checks native
``torch.xpu`` first and only falls back to importing IPEX if that alone
isn't enough, so it works correctly on both old and new PyTorch.

Hardware: Intel Data Center GPU Max 1550, Flex 170; Arc A770
Runtime: oneAPI (Level Zero, oneMKL, oneDNN)
Collective: oneCCL (Intel oneAPI Collective Communications Library)
Adapter: native torch.xpu (>=2.9) or intel-extension-for-pytorch (<2.9)

STATUS: BACKEND STUB
--------------------
Detection logic is implemented.  Ecosystem patches are NOT yet implemented.
"""

from __future__ import annotations

from ascend_compat.backends.registry import BackendInfo


def _ensure_xpu_registered() -> bool:
    """Make sure ``torch.xpu`` is registered, importing IPEX if needed.

    On PyTorch >=2.9, ``torch.xpu`` is built in and needs no extra import.
    On older PyTorch, the device is only registered once
    ``intel_extension_for_pytorch`` has been imported.  Try native first so
    we don't require a discontinued package on modern PyTorch.
    """
    import torch

    if hasattr(torch, "xpu"):
        return True
    try:
        import intel_extension_for_pytorch  # type: ignore[import-untyped]  # noqa: F401
    except Exception:
        return False
    return hasattr(torch, "xpu")


class IntelBackend(BackendInfo):
    """Intel GPU via native torch.xpu (falls back to legacy IPEX import)."""

    name = "intel"
    device_type = "xpu"
    adapter_module = "intel_extension_for_pytorch"
    collective_backend = "ccl"
    visible_devices_env = "ZE_AFFINITY_MASK"
    display_name = "Intel GPU (XPU)"
    docs_url = "https://github.com/intel/intel-extension-for-pytorch"

    @staticmethod
    def is_available() -> bool:
        """Check if Intel GPU hardware is present and torch.xpu is registered."""
        try:
            import torch

            return _ensure_xpu_registered() and torch.xpu.is_available()
        except Exception:
            return False

    @staticmethod
    def device_count() -> int:
        """Return number of Intel GPU devices."""
        try:
            import torch

            if _ensure_xpu_registered():
                return torch.xpu.device_count()
        except Exception:
            pass
        return 0

    @staticmethod
    def get_device_name(index: int = 0) -> str:
        """Return the Intel GPU model name."""
        try:
            import torch

            if _ensure_xpu_registered():
                return torch.xpu.get_device_name(index)
        except Exception:
            pass
        return "Intel GPU (unknown model)"

    @classmethod
    def get_adapter_version(cls) -> str | None:
        """Return the IPEX version if present, else the native torch version."""
        try:
            import intel_extension_for_pytorch as ipex  # type: ignore[import-untyped]

            return getattr(ipex, "__version__", "unknown")
        except ImportError:
            pass
        try:
            import torch

            if hasattr(torch, "xpu"):
                return f"native (torch {torch.__version__})"
        except Exception:
            pass
        return None
