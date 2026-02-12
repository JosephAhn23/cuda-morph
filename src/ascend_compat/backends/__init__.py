"""Pluggable backend registry for domestic AI chips.

Each backend module implements a common protocol that the core shim uses
to translate ``torch.cuda`` calls to the appropriate vendor API.

Adding a new backend:
1. Create a module in this package (e.g. ``biren.py``)
2. Define a class implementing the :class:`BackendInfo` protocol
3. Register it in :data:`BACKEND_REGISTRY`

The core shim (``cuda_shim/_monkey_patch.py``) reads from the active
backend to know:
- What device type to translate ``"cuda"`` to (``"npu"``, ``"mlu"``, etc.)
- What collective backend replaces NCCL
- What environment variable replaces ``CUDA_VISIBLE_DEVICES``
"""

from __future__ import annotations

from typing import Dict, Type

from ascend_compat.backends.registry import BackendInfo
from ascend_compat.backends.ascend import AscendBackend
from ascend_compat.backends.cambricon import CambriconBackend

# All known backends, keyed by their short name.
# Detection priority follows insertion order.
BACKEND_REGISTRY: Dict[str, Type[BackendInfo]] = {
    "ascend": AscendBackend,
    "cambricon": CambriconBackend,
}

__all__ = ["BACKEND_REGISTRY", "BackendInfo", "AscendBackend", "CambriconBackend"]
