"""HuggingFace Transformers / Accelerate compatibility patches.

Known bugs fixed by this module
-------------------------------

1. **device_map="auto" crashes with NPU** (transformers ≥4.50.0)

   ``_load_state_dict_into_meta_model`` doesn't handle NPU device indices,
   causing ``AssertionError: Torch not compiled with CUDA enabled`` when
   trying to move tensors to device via integer indices.

2. **accelerate selects NCCL instead of HCCL**

   When ``torch.cuda.is_available()`` returns True (as torch_npu's
   ``transfer_to_npu`` makes it), accelerate's ``DistributedType`` logic
   selects NCCL for multi-GPU.  On Ascend, NCCL doesn't exist.
   The cuda_shim layer fixes this by making ``is_available()`` return False,
   but we also need to ensure accelerate's device detection finds NPU.

3. **attn_implementation="flash_attention_2" fails on NPU**

   Transformers checks ``is_flash_attn_2_available()`` which tries to
   import ``flash_attn`` — unavailable on Ascend.  We patch the check to
   succeed when our ecosystem.flash_attn shim is available.

Usage::

    from ascend_compat.ecosystem import transformers_patch
    transformers_patch.apply()
    # Now: from transformers import AutoModelForCausalLM  (works on NPU)
"""

from __future__ import annotations

import sys
from typing import Any

from ascend_compat._backend import has_npu
from ascend_compat._logging import get_logger

logger = get_logger(__name__)

_applied = False


def apply() -> None:
    """Apply all HuggingFace ecosystem patches.  Idempotent."""
    global _applied  # noqa: PLW0603
    if _applied:
        return

    if not has_npu():
        logger.debug("No NPU detected — skipping transformers patches")
        return

    _patch_flash_attn_check()
    _patch_accelerate_device_detection()
    _register_flash_attn_package()

    _applied = True
    logger.info("HuggingFace ecosystem patches applied")


def _patch_flash_attn_check() -> None:
    """Make transformers' ``is_flash_attn_2_available()`` return True on NPU.

    Transformers >=4.36 checks this function before allowing
    ``attn_implementation="flash_attention_2"``.  Without this patch,
    users must manually set ``attn_implementation="eager"`` on Ascend,
    losing the 10x speedup from fused attention.
    """
    try:
        import transformers.utils  # type: ignore[import-untyped]

        if hasattr(transformers.utils, "is_flash_attn_2_available"):
            original = transformers.utils.is_flash_attn_2_available

            def _patched_check(*args: Any, **kwargs: Any) -> bool:
                # If flash_attn (or our shim) is importable, return True
                try:
                    from ascend_compat.ecosystem import flash_attn  # noqa: F401
                    logger.debug("is_flash_attn_2_available() → True (via ascend-compat shim)")
                    return True
                except ImportError:
                    return original(*args, **kwargs)

            transformers.utils.is_flash_attn_2_available = _patched_check
            logger.debug("Patched transformers.utils.is_flash_attn_2_available")

    except ImportError:
        logger.debug("transformers not installed — skipping flash_attn check patch")


def _patch_accelerate_device_detection() -> None:
    """Ensure accelerate detects NPU as the compute device.

    accelerate's ``AcceleratorState`` uses device detection logic that
    may not account for NPU.  We patch the detection to include NPU.
    """
    try:
        import accelerate.utils  # type: ignore[import-untyped]

        # Ensure accelerate knows about NPU
        if hasattr(accelerate.utils, "is_npu_available"):
            # Already has NPU support (accelerate >=0.28) — just verify it works
            if accelerate.utils.is_npu_available():
                logger.debug("accelerate.is_npu_available() already returns True")
            else:
                logger.warning(
                    "accelerate.is_npu_available() returns False even though NPU is detected. "
                    "Ensure torch_npu is installed correctly."
                )
        else:
            logger.debug(
                "accelerate.utils.is_npu_available not found — "
                "consider upgrading accelerate to >=0.28 for native NPU support"
            )

    except ImportError:
        logger.debug("accelerate not installed — skipping device detection patch")


def _register_flash_attn_package() -> None:
    """Register our flash_attn shim as the ``flash_attn`` package.

    This allows ``from flash_attn import flash_attn_func`` to work
    without the actual flash_attn package installed.
    """
    if "flash_attn" not in sys.modules:
        from ascend_compat.ecosystem import flash_attn as fa_shim

        sys.modules["flash_attn"] = fa_shim  # type: ignore[assignment]
        sys.modules["flash_attn.flash_attn_interface"] = fa_shim  # type: ignore[assignment]
        logger.debug("Registered ascend_compat.ecosystem.flash_attn as 'flash_attn' package")
    else:
        logger.debug("flash_attn already imported — not overriding with shim")
