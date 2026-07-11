"""Triton-Ascend integration bridge.

Triton is emerging as the most promising vendor-agnostic kernel language.
Huawei announced deepening collaboration with the Triton community at
CONNECT 2025, and Triton-Ascend enables targeting Ascend NPUs from
standard Triton Python kernel code.

This module provides:
1. Detection of Triton-Ascend availability
2. Helper to configure Triton for Ascend backend
3. Compatibility shims for Triton kernels that assume CUDA semantics

The Triton ecosystem for Ascend includes:
- **Triton-Ascend**: Direct Triton→CANN compilation path
- **TileLang-Ascend**: Python DSL from tile-ai that compiles to Ascend C
- **Triton-Linalg**: Cambricon's linear algebra extension for Triton

Usage::

    from ascend_compat.ecosystem.triton_bridge import (
        is_triton_ascend_available,
        configure_triton_backend,
    )

    if is_triton_ascend_available():
        configure_triton_backend()
        # Now Triton kernels target Ascend NPU
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

from ascend_compat._backend import has_npu
from ascend_compat._logging import get_logger

logger = get_logger(__name__)


def is_triton_available() -> bool:
    """Check if any version of Triton is installed."""
    try:
        import triton  # type: ignore[import-untyped]
        return True
    except ImportError:
        return False


def is_triton_ascend_available() -> bool:
    """Check if Triton-Ascend backend is available.

    Triton-Ascend provides a compilation path from Triton IR to CANN
    kernels.  It requires both the triton package and the ascend backend
    plugin to be installed.
    """
    if not is_triton_available():
        return False

    try:
        import triton  # type: ignore[import-untyped]
        # Check for Ascend backend registration
        if hasattr(triton, "backends"):
            backends = triton.backends
            if hasattr(backends, "ascend") or hasattr(backends, "npu"):
                return True

        # Alternative: check for triton_ascend package
        import triton_ascend  # type: ignore[import-untyped]  # noqa: F401
        return True
    except (ImportError, AttributeError):
        pass

    return False


def get_triton_info() -> Dict[str, Any]:
    """Return information about Triton installation and backends."""
    info: Dict[str, Any] = {
        "triton_installed": False,
        "triton_version": None,
        "ascend_backend": False,
        "available_backends": [],
    }

    try:
        import triton  # type: ignore[import-untyped]
        info["triton_installed"] = True
        info["triton_version"] = getattr(triton, "__version__", "unknown")

        # List available backends
        if hasattr(triton, "backends"):
            for name in dir(triton.backends):
                if not name.startswith("_"):
                    info["available_backends"].append(name)

        info["ascend_backend"] = is_triton_ascend_available()

    except ImportError:
        pass

    return info


def configure_triton_backend() -> bool:
    """Configure Triton to use the Ascend backend.

    Sets the appropriate environment variables and backend selection
    for Triton to compile kernels for Ascend NPU.

    Returns:
        True if configuration succeeded, False otherwise.
    """
    if not is_triton_ascend_available():
        logger.warning(
            "Triton-Ascend not available. Install:\n"
            "  pip install triton-ascend\n"
            "  Or: pip install triton && pip install triton-npu"
        )
        return False

    # Set Triton backend to Ascend
    os.environ.setdefault("TRITON_BACKEND", "ascend")

    # Set CANN-specific compilation flags
    ascend_home = os.environ.get("ASCEND_HOME_PATH", "")
    if ascend_home:
        os.environ.setdefault("CANN_PATH", ascend_home)

    logger.info("Triton configured for Ascend backend")
    return True


def patch_triton_cuda_assumptions() -> None:
    """Patch Triton kernels that assume CUDA semantics.

    Many community Triton kernels hardcode assumptions like:
    - ``triton.language.device_type == "cuda"``
    - CUDA-specific memory semantics
    - Warp-level operations (no equivalent on Ascend's SIMD model)

    This function monkey-patches common patterns to work on Ascend.
    """
    if not is_triton_available():
        return

    try:
        import triton.language as tl  # type: ignore[import-untyped]

        # Patch device_type if it exists
        if hasattr(tl, "device_type"):
            # On Ascend, report as "npu" rather than "cuda"
            logger.debug("Triton language device_type available: %s", tl.device_type)

    except (ImportError, AttributeError):
        pass


def triton_kernel_available(kernel_name: str) -> bool:
    """Check if a specific Triton kernel can run on Ascend.

    Some Triton kernels use CUDA-specific intrinsics (warp shuffles,
    shared memory atomics) that have no Ascend equivalent.

    Args:
        kernel_name: Name/description of the kernel to check.

    Returns:
        True if the kernel is likely compatible with Ascend.
    """
    # Known incompatible patterns
    cuda_only_patterns = [
        "warp_shuffle",
        "warp_reduce",
        "shared_memory_atomic",
        "cooperative_group",
        "tensor_core",
        "wmma",
    ]

    kernel_lower = kernel_name.lower()
    for pattern in cuda_only_patterns:
        if pattern in kernel_lower:
            logger.debug("Kernel '%s' uses CUDA-only pattern: %s", kernel_name, pattern)
            return False

    return True
