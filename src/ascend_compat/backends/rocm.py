"""AMD ROCm backend (PyTorch ROCm build).

AMD's ROCm (Radeon Open Compute) platform is the primary open-source
alternative to NVIDIA CUDA for GPU computing.  PyTorch ships official
ROCm builds that register as ``"cuda"`` device type (via HIP translation),
but with ROCm-specific behavior under the hood.

Unlike domestic Chinese backends which use PrivateUse1, ROCm uses PyTorch's
native CUDA backend with HIP translation.  This means:
- ``torch.cuda.is_available()`` returns True on ROCm
- ``torch.device("cuda")`` works natively
- Most CUDA code runs without modification

**So why does cuda-morph need a ROCm backend?**

Because the ecosystem still breaks:
1. ``flash-attn`` requires NVIDIA CUDA to compile (not ROCm)
2. Some libraries check ``torch.version.cuda`` and refuse to run on ROCm
3. Custom CUDA extensions fail to compile without NVCC
4. NCCL initialization differs from RCCL on multi-node setups

cuda-morph's ROCm backend:
- Detects ROCm vs CUDA correctly (via ``torch.version.hip``)
- Routes flash_attn to ROCm-compatible alternatives (Triton flash-attn, CK)
- Provides RCCL collective backend mapping
- Translates NVIDIA-specific error messages

Hardware: AMD Instinct MI210, MI250X, MI300X; Radeon RX 7900 XTX
Runtime: ROCm (HIP, rocBLAS, MIOpen)
Collective: RCCL (ROCm Communication Collectives Library)
Adapter: PyTorch ROCm build (official — pip install torch --index-url ...)

STATUS: BACKEND STUB
--------------------
Detection logic is implemented.  Ecosystem patches are NOT yet implemented.
"""

from __future__ import annotations

from ascend_compat.backends.registry import BackendInfo


class ROCmBackend(BackendInfo):
    """AMD ROCm via PyTorch ROCm build."""

    name = "rocm"
    device_type = "cuda"  # ROCm presents as CUDA device via HIP
    adapter_module = "torch"  # No separate adapter — built into PyTorch
    collective_backend = "rccl"
    visible_devices_env = "HIP_VISIBLE_DEVICES"
    display_name = "AMD ROCm"
    docs_url = "https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"

    @staticmethod
    def is_available() -> bool:
        """Check if AMD ROCm hardware is present.

        ROCm registers as CUDA in PyTorch, so we detect it by checking
        for ``torch.version.hip`` which is set in ROCm builds.
        """
        try:
            import torch
            # ROCm builds set torch.version.hip to a version string
            # CUDA builds have torch.version.hip as None
            hip_version = getattr(torch.version, "hip", None)
            if hip_version is not None and torch.cuda.is_available():
                return True
        except Exception:
            pass
        return False

    @staticmethod
    def device_count() -> int:
        """Return number of AMD GPUs."""
        try:
            import torch
            hip_version = getattr(torch.version, "hip", None)
            if hip_version is not None:
                return torch.cuda.device_count()
        except Exception:
            pass
        return 0

    @staticmethod
    def get_device_name(index: int = 0) -> str:
        """Return the AMD GPU model name."""
        try:
            import torch
            hip_version = getattr(torch.version, "hip", None)
            if hip_version is not None and torch.cuda.is_available():
                return torch.cuda.get_device_name(index)
        except Exception:
            pass
        return "AMD GPU (unknown model)"

    @classmethod
    def get_adapter_version(cls) -> str | None:
        """Return the ROCm/HIP version."""
        try:
            import torch
            hip_version = getattr(torch.version, "hip", None)
            if hip_version is not None:
                return f"ROCm {hip_version}"
        except ImportError:
            pass
        return None
