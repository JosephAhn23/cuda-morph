"""DeepSpeed compatibility patches for Ascend NPU.

Known bugs fixed by this module
-------------------------------

1. **HCCL backend not registered**

   DeepSpeed's ``init_distributed`` defaults to NCCL when
   ``torch.cuda.is_available()`` is True.  On Ascend, HCCL must be used.
   Since our cuda_shim makes ``is_available()`` return False, DeepSpeed
   falls through to CPU/Gloo.  This patch registers HCCL explicitly.

2. **timer.py stream synchronization bug**

   DeepSpeed's ``SynchronizedWallClockTimer`` uses ``torch.cuda.Event``
   for timing.  On NPU, this needs to use ``torch.npu.Event`` instead.
   Without the patch, timer calls crash during training.

3. **ASCEND_RT_VISIBLE_DEVICES environment variable**

   DeepSpeed reads ``CUDA_VISIBLE_DEVICES`` to determine device assignment.
   On Ascend, the equivalent is ``ASCEND_RT_VISIBLE_DEVICES``.

Usage::

    from ascend_compat.ecosystem import deepspeed_patch
    deepspeed_patch.apply()
    # Now: import deepspeed  (works on NPU)
"""

from __future__ import annotations

import os
from typing import Any

from ascend_compat._backend import has_npu
from ascend_compat._logging import get_logger

logger = get_logger(__name__)

_applied = False


def apply() -> None:
    """Apply all DeepSpeed patches for Ascend.  Idempotent."""
    global _applied  # noqa: PLW0603
    if _applied:
        return

    if not has_npu():
        logger.debug("No NPU detected — skipping DeepSpeed patches")
        return

    _patch_visible_devices_env()
    _patch_deepspeed_dist_backend()
    _patch_deepspeed_timer()

    _applied = True
    logger.info("DeepSpeed compatibility patches applied")


def _patch_visible_devices_env() -> None:
    """Map CUDA_VISIBLE_DEVICES → ASCEND_RT_VISIBLE_DEVICES if needed.

    Many launch scripts set CUDA_VISIBLE_DEVICES.  On Ascend, the
    driver reads ASCEND_RT_VISIBLE_DEVICES instead.
    """
    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    ascend_vis = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")

    if cuda_vis and not ascend_vis:
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = cuda_vis
        logger.info(
            "Mapped CUDA_VISIBLE_DEVICES=%s → ASCEND_RT_VISIBLE_DEVICES", cuda_vis
        )


def _patch_deepspeed_dist_backend() -> None:
    """Ensure DeepSpeed uses HCCL for distributed training on Ascend."""
    try:
        import deepspeed  # type: ignore[import-untyped]

        if hasattr(deepspeed, "comm") and hasattr(deepspeed.comm, "init_distributed"):
            original_init = deepspeed.comm.init_distributed

            def _patched_init(
                dist_backend: str = "nccl",
                **kwargs: Any,
            ) -> Any:
                if dist_backend == "nccl":
                    dist_backend = "hccl"
                    logger.info("DeepSpeed dist_backend: nccl → hccl (Ascend)")
                return original_init(dist_backend=dist_backend, **kwargs)

            deepspeed.comm.init_distributed = _patched_init
            logger.debug("Patched deepspeed.comm.init_distributed")

    except ImportError:
        logger.debug("DeepSpeed not installed — skipping dist backend patch")


def _patch_deepspeed_timer() -> None:
    """Fix DeepSpeed's SynchronizedWallClockTimer to use NPU events.

    DeepSpeed's timer module creates ``torch.cuda.Event`` objects directly.
    On NPU, these need to be ``torch.npu.Event`` instead.
    """
    try:
        import deepspeed.utils.timer as ds_timer  # type: ignore[import-untyped]
        import torch

        if not hasattr(torch, "npu"):
            return

        if hasattr(ds_timer, "SynchronizedWallClockTimer"):
            timer_cls = ds_timer.SynchronizedWallClockTimer

            if hasattr(timer_cls, "_start_event"):
                # Already patched or different version
                return

            original_init = timer_cls.__init__

            def _patched_timer_init(self: Any, *args: Any, **kwargs: Any) -> None:
                original_init(self, *args, **kwargs)
                # Replace CUDA events with NPU events
                if hasattr(self, "start_event") and hasattr(torch.npu, "Event"):
                    self.start_event = torch.npu.Event(enable_timing=True)
                    self.end_event = torch.npu.Event(enable_timing=True)

            timer_cls.__init__ = _patched_timer_init
            logger.debug("Patched DeepSpeed SynchronizedWallClockTimer")

    except ImportError:
        logger.debug("DeepSpeed timer module not found — skipping timer patch")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to patch DeepSpeed timer: %s", exc)
