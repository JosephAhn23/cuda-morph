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
import warnings
from typing import Any, Dict, Optional, Tuple

from ascend_compat._backend import has_npu
from ascend_compat._logging import get_logger

logger = get_logger(__name__)

_applied = False
_patch_results: Dict[str, bool] = {}

# Tested DeepSpeed versions
_TESTED_DEEPSPEED = ((0, 12), (0, 13), (0, 14), (0, 15), (0, 16))


def _get_deepspeed_version() -> Optional[Tuple[int, ...]]:
    """Return DeepSpeed (major, minor) version or None."""
    try:
        import deepspeed  # type: ignore[import-untyped]
        ver = getattr(deepspeed, "__version__", "0.0.0")
        parts = ver.split(".")[:2]
        return tuple(int(p) for p in parts)
    except (ImportError, ValueError):
        return None


def apply() -> None:
    """Apply all DeepSpeed patches for Ascend.  Idempotent."""
    global _applied  # noqa: PLW0603
    if _applied:
        return

    if not has_npu():
        logger.debug("No NPU detected — skipping DeepSpeed patches")
        return

    # Version guard
    ds_ver = _get_deepspeed_version()
    if ds_ver and ds_ver[:2] not in _TESTED_DEEPSPEED:
        tested_strs = [f"{v[0]}.{v[1]}" for v in _TESTED_DEEPSPEED]
        warnings.warn(
            f"DeepSpeed {ds_ver[0]}.{ds_ver[1]} has not been tested with ascend-compat. "
            f"Tested: {', '.join(tested_strs)}. Patches may not work correctly.",
            FutureWarning,
            stacklevel=2,
        )

    _patch_visible_devices_env()
    _patch_deepspeed_dist_backend()
    _patch_deepspeed_timer()

    _applied = True
    logger.info("DeepSpeed compatibility patches applied")


def get_patch_results() -> Dict[str, bool]:
    """Return verification results: {patch_name: landed_successfully}."""
    return dict(_patch_results)


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
            # Verify patch landed
            _patch_results["dist_backend"] = (
                deepspeed.comm.init_distributed is _patched_init
            )
            logger.debug("Patched deepspeed.comm.init_distributed")
        else:
            _patch_results["dist_backend"] = False
            logger.warning(
                "deepspeed.comm.init_distributed not found. "
                "DeepSpeed API may have changed — HCCL backend patch skipped."
            )

    except ImportError:
        _patch_results["dist_backend"] = False
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
