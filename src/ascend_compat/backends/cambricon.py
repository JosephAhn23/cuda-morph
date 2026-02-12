"""Cambricon MLU backend (torch_mlu + BANG).

Cambricon produces the MLU370 and MLU590 series AI accelerators,
widely deployed in China's domestic AI infrastructure.  Their PyTorch
adapter is ``torch_mlu``, which registers the ``"mlu"`` device type
via PyTorch's PrivateUse1 mechanism.

Hardware: MLU370-S4, MLU370-X8, MLU590
Runtime: BANG (Basic Architecture for Next Generation)
Collective: CNCL (Cambricon Collective Communication Library)
Adapter: torch_mlu (https://github.com/Cambricon/torch_mlu)

STATUS: BACKEND STUB
--------------------
This module implements detection and configuration for Cambricon MLU.
The ecosystem patches (flash_attn mapping, DeepSpeed CNCL, etc.) are
NOT yet implemented — they require MLU hardware for development and
validation.

What IS implemented:
- Hardware detection via torch_mlu
- Device type mapping ("cuda" → "mlu")
- Collective backend mapping (NCCL → CNCL)
- Visible devices env mapping (CUDA_VISIBLE_DEVICES → MLU_VISIBLE_DEVICES)

What is NOT implemented:
- flash_attn equivalent for MLU (Cambricon's fused attention API differs)
- DeepSpeed CNCL integration
- vLLM MLU attention routing
- BANG error code translation
- Operator verification on MLU hardware

To contribute MLU support, see STRATEGY.md for partnership details.
"""

from __future__ import annotations

from ascend_compat.backends.registry import BackendInfo


class CambriconBackend(BackendInfo):
    """Cambricon MLU via torch_mlu."""

    name = "cambricon"
    device_type = "mlu"
    adapter_module = "torch_mlu"
    collective_backend = "cncl"
    visible_devices_env = "MLU_VISIBLE_DEVICES"
    display_name = "Cambricon MLU"
    docs_url = "https://github.com/Cambricon/torch_mlu"

    @staticmethod
    def is_available() -> bool:
        """Check if Cambricon MLU hardware is present and torch_mlu works."""
        try:
            import torch_mlu  # type: ignore[import-untyped]  # noqa: F401
            import torch
            return hasattr(torch, "mlu") and torch.mlu.is_available()
        except Exception:
            return False

    @staticmethod
    def device_count() -> int:
        """Return number of Cambricon MLU devices."""
        try:
            import torch
            if hasattr(torch, "mlu"):
                return torch.mlu.device_count()
        except Exception:
            pass
        return 0

    @staticmethod
    def get_device_name(index: int = 0) -> str:
        """Return the Cambricon MLU model name."""
        try:
            import torch
            if hasattr(torch, "mlu"):
                return torch.mlu.get_device_name(index)
        except Exception:
            pass
        return "Cambricon MLU (unknown model)"
