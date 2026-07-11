"""Huawei Ascend NPU backend (torch_npu + CANN).

This is the primary backend for cuda-morph.  It translates
``torch.cuda`` calls to ``torch.npu`` equivalents using Huawei's
official ``torch_npu`` adapter.

Hardware: Ascend 910A, 910B, 310P
Runtime: CANN (Compute Architecture for Neural Networks)
Collective: HCCL (Huawei Collective Communication Library)
Adapter: torch_npu (https://gitee.com/ascend/pytorch)
"""

from __future__ import annotations

from typing import Optional

from ascend_compat.backends.registry import BackendInfo


class AscendBackend(BackendInfo):
    """Huawei Ascend NPU via torch_npu."""

    name = "ascend"
    device_type = "npu"
    adapter_module = "torch_npu"
    collective_backend = "hccl"
    visible_devices_env = "ASCEND_RT_VISIBLE_DEVICES"
    display_name = "Huawei Ascend NPU"
    docs_url = "https://gitee.com/ascend/pytorch"

    @staticmethod
    def is_available() -> bool:
        """Check if Ascend NPU hardware is present and torch_npu works."""
        try:
            import torch_npu  # type: ignore[import-untyped]  # noqa: F401
            import torch
            return hasattr(torch, "npu") and torch.npu.is_available()
        except Exception:
            return False

    @staticmethod
    def device_count() -> int:
        """Return number of Ascend NPU devices."""
        try:
            import torch
            if hasattr(torch, "npu"):
                return torch.npu.device_count()
        except Exception:
            pass
        return 0

    @staticmethod
    def get_device_name(index: int = 0) -> str:
        """Return the Ascend NPU model name."""
        try:
            import torch
            if hasattr(torch, "npu"):
                return torch.npu.get_device_name(index)
        except Exception:
            pass
        return "Ascend NPU (unknown model)"
