"""Layer 2: Core CUDA → Ascend compatibility shim.

This package intercepts torch.cuda API calls and routes them to torch.npu
equivalents provided by torch_npu.  It operates at the Python level only —
all C++/CANN integration is handled by torch_npu's PrivateUse1 backend.

Architecture:
    _registry.py      — Version-aware mapping of torch.cuda attrs → torch.npu
    _import_hook.py   — sys.meta_path interceptor for `import torch.cuda` 
    _monkey_patch.py  — Runtime patching of torch.cuda, torch.device, Tensor.cuda()

Critical design decision:
    torch.cuda.is_available() → False on Ascend systems.

    This seems counterintuitive, but it fixes the most pernicious bug in the
    ecosystem: when is_available() returns True, HuggingFace accelerate selects
    the NCCL backend instead of HCCL, causing silent distributed training
    failures.  torch_npu's transfer_to_npu makes the same mistake.  By returning
    False, we force libraries to check the NPU path instead.
"""

from ascend_compat.cuda_shim._monkey_patch import (
    activate,
    deactivate,
    get_all_patch_stats,
    get_patch_names,
    get_patch_stats,
    is_activated,
    reset_patch_stats,
)

__all__ = [
    "activate",
    "deactivate",
    "is_activated",
    "get_patch_stats",
    "get_all_patch_stats",
    "get_patch_names",
    "reset_patch_stats",
]
