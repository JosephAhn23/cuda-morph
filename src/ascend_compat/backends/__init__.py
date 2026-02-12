"""Pluggable backend registry for non-NVIDIA accelerators.

cuda-morph supports any PyTorch backend — domestic Chinese chips (Ascend,
Cambricon) and global alternatives (AMD ROCm, Intel XPU).  Each backend
module implements a common protocol that the core shim uses to translate
``torch.cuda`` calls to the appropriate vendor API.

Adding a new backend:
1. Create a module in this package (e.g. ``biren.py``)
2. Define a class implementing the :class:`BackendInfo` protocol
3. Register it in :data:`BACKEND_REGISTRY`

The core shim (``cuda_shim/_monkey_patch.py``) reads from the active
backend to know:
- What device type to translate ``"cuda"`` to (``"npu"``, ``"mlu"``, ``"xpu"``, etc.)
- What collective backend replaces NCCL
- What environment variable replaces ``CUDA_VISIBLE_DEVICES``
"""

from __future__ import annotations

from typing import Dict, Type

from ascend_compat.backends.registry import BackendInfo
from ascend_compat.backends.ascend import AscendBackend
from ascend_compat.backends.cambricon import CambriconBackend
from ascend_compat.backends.rocm import ROCmBackend
from ascend_compat.backends.intel import IntelBackend

# All known backends, keyed by their short name.
# Detection priority follows insertion order:
#   1. Domestic Chinese backends (most likely to need shim)
#   2. Global alternatives (AMD, Intel)
# NVIDIA CUDA and CPU are handled directly by _backend.py, not here.
BACKEND_REGISTRY: Dict[str, Type[BackendInfo]] = {
    "ascend": AscendBackend,
    "cambricon": CambriconBackend,
    "rocm": ROCmBackend,
    "intel": IntelBackend,
}

__all__ = [
    "BACKEND_REGISTRY",
    "BackendInfo",
    "AscendBackend",
    "CambriconBackend",
    "ROCmBackend",
    "IntelBackend",
]
