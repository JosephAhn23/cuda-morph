"""Automatic dtype management for Ascend NPU hardware constraints.

Ascend Da Vinci architecture has specific dtype limitations:

- **Ascend 910A**: Cube Unit only supports FP16 matrix multiply.  FP32 GEMM
  is emulated via 3-pass FP16 decomposition (77% of theoretical FP32 peak).
  BF16 support varies by firmware version.

- **Ascend 910B**: Native FP32 Cube at 73.73 TFLOPS, FP16 at 320 TFLOPS.
  BF16 support via firmware update.

- **Ascend 910C**: ~800 TFLOPS FP16 (dual-die), FP32 improved.

- **ALL models**: No FP64 support (hardware limitation).

This module provides:

1. ``patch_dtype_creation()`` — Intercepts ``torch.tensor()`` and
   ``torch.zeros/ones/randn()`` to substitute unsupported dtypes.

2. ``DTypePolicy`` — Configurable policy for automatic dtype substitution.

3. ``check_dtype_support()`` — Query whether a specific dtype is natively
   supported on the current hardware.

Usage::

    from ascend_compat.cuda_shim.dtype_manager import apply_dtype_policy, DTypePolicy

    # Auto-substitute bf16 → fp16 and fp64 → fp32
    apply_dtype_policy(DTypePolicy.AUTO)
"""

from __future__ import annotations

import enum
import functools
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ascend_compat._logging import get_logger

logger = get_logger(__name__)


class DTypePolicy(enum.Enum):
    """Policy for automatic dtype substitution."""

    STRICT = "strict"      # Raise error on unsupported dtype
    AUTO = "auto"          # Silently substitute to best supported dtype
    WARN = "warn"          # Substitute but emit a warning
    DISABLED = "disabled"  # No dtype management


# Dtype substitution map: unsupported → best supported equivalent
# This is conservative: only substitute when we KNOW the dtype is unsupported
_DTYPE_SUBSTITUTIONS: Dict[str, str] = {
    "float64": "float32",     # No FP64 on any Ascend model
    "float16": "float16",     # Native — no change needed
    "float32": "float32",     # Native on 910B+, emulated on 910A
}

# Whether BF16 needs substitution depends on hardware.
# On 910A without firmware update: bfloat16 → float16
# On 910B+: bfloat16 is native
_BF16_NEEDS_SUBSTITUTION: Optional[bool] = None

_active_policy = DTypePolicy.DISABLED
_patched = False


# ---------------------------------------------------------------------------
# Hardware dtype probing
# ---------------------------------------------------------------------------


def _probe_bf16_support() -> bool:
    """Check if the current NPU hardware supports bfloat16 natively.

    Tests by creating a small BF16 tensor on the NPU.  If it fails,
    BF16 needs substitution.
    """
    global _BF16_NEEDS_SUBSTITUTION  # noqa: PLW0603

    if _BF16_NEEDS_SUBSTITUTION is not None:
        return not _BF16_NEEDS_SUBSTITUTION

    try:
        import torch
        if hasattr(torch, "npu") and torch.npu.is_available():
            # Try to create a BF16 tensor on NPU
            t = torch.zeros(2, 2, dtype=torch.bfloat16, device="npu:0")
            # Try a matmul to test Cube Unit support
            _ = torch.matmul(t, t)
            _BF16_NEEDS_SUBSTITUTION = False
            logger.debug("BF16 is natively supported on this NPU")
            return True
    except Exception as exc:
        logger.debug("BF16 not supported on this NPU: %s", exc)
        _BF16_NEEDS_SUBSTITUTION = True
        return False

    # No NPU available — assume BF16 is fine (will use CPU anyway)
    _BF16_NEEDS_SUBSTITUTION = False
    return True


def check_dtype_support(dtype_name: str) -> Tuple[bool, Optional[str]]:
    """Check if a dtype is natively supported on the current NPU.

    Args:
        dtype_name: PyTorch dtype name (e.g. "float64", "bfloat16").

    Returns:
        Tuple of (is_supported, substitute_name_or_None).
    """
    if dtype_name == "float64":
        return False, "float32"

    if dtype_name == "bfloat16":
        if _probe_bf16_support():
            return True, None
        return False, "float16"

    if dtype_name in ("float32", "float16"):
        return True, None

    # Integer types are generally supported
    if dtype_name in ("int8", "int16", "int32", "int64", "uint8", "bool"):
        return True, None

    # Quantized types are unsupported
    if "quint" in dtype_name or "qint" in dtype_name:
        return False, None

    return True, None  # Unknown — assume supported


def get_substitution_map() -> Dict[str, str]:
    """Return the active dtype substitution map.

    Includes bf16 substitution if hardware doesn't support it.
    """
    subs = dict(_DTYPE_SUBSTITUTIONS)
    if _BF16_NEEDS_SUBSTITUTION or (_BF16_NEEDS_SUBSTITUTION is None and not _probe_bf16_support()):
        subs["bfloat16"] = "float16"
        logger.debug("BF16 → FP16 substitution active")
    return subs


# ---------------------------------------------------------------------------
# Tensor creation patching
# ---------------------------------------------------------------------------


def apply_dtype_policy(policy: DTypePolicy = DTypePolicy.WARN) -> None:
    """Apply a dtype policy for automatic substitution.

    Args:
        policy: How to handle unsupported dtypes.
            - ``STRICT``: Raise RuntimeError
            - ``AUTO``: Silently substitute
            - ``WARN``: Substitute with warning (default)
            - ``DISABLED``: No substitution

    This patches tensor creation functions (``torch.tensor``, ``torch.zeros``,
    ``torch.ones``, ``torch.randn``, ``torch.empty``, ``torch.full``)
    to intercept the ``dtype`` argument.
    """
    global _active_policy, _patched  # noqa: PLW0603

    _active_policy = policy

    if policy == DTypePolicy.DISABLED:
        if _patched:
            _unpatch_creation_fns()
        return

    if not _patched:
        _patch_creation_fns()


def _resolve_dtype(dtype: Any) -> Any:
    """Resolve a potentially unsupported dtype to a supported one.

    Args:
        dtype: A torch.dtype or None.

    Returns:
        The (possibly substituted) dtype.
    """
    if dtype is None or _active_policy == DTypePolicy.DISABLED:
        return dtype

    try:
        import torch
    except ImportError:
        return dtype

    # Map torch.dtype to string name
    dtype_name = _dtype_to_name(dtype)
    if dtype_name is None:
        return dtype

    is_supported, substitute = check_dtype_support(dtype_name)

    if is_supported:
        return dtype

    if substitute is None:
        if _active_policy == DTypePolicy.STRICT:
            raise RuntimeError(
                f"Dtype {dtype_name} is not supported on Ascend NPU and has no substitute. "
                f"Use a different dtype."
            )
        return dtype

    substitute_dtype = getattr(torch, substitute, dtype)

    if _active_policy == DTypePolicy.STRICT:
        raise RuntimeError(
            f"Dtype {dtype_name} is not supported on Ascend NPU. "
            f"Suggested substitute: torch.{substitute}"
        )
    elif _active_policy == DTypePolicy.WARN:
        import warnings
        warnings.warn(
            f"ascend-compat: Substituting torch.{dtype_name} → torch.{substitute} "
            f"(unsupported on Ascend NPU)",
            UserWarning,
            stacklevel=4,
        )

    logger.debug("dtype substitution: torch.%s → torch.%s", dtype_name, substitute)
    return substitute_dtype


def _dtype_to_name(dtype: Any) -> Optional[str]:
    """Convert a torch.dtype to its string name."""
    try:
        import torch
        _DTYPE_NAMES = {
            torch.float16: "float16",
            torch.float32: "float32",
            torch.float64: "float64",
            torch.bfloat16: "bfloat16",
            torch.int8: "int8",
            torch.int16: "int16",
            torch.int32: "int32",
            torch.int64: "int64",
            torch.uint8: "uint8",
            torch.bool: "bool",
        }
        return _DTYPE_NAMES.get(dtype)
    except (ImportError, AttributeError):
        return None


# Creation function wrappers
_originals_creation: Dict[str, Any] = {}


def _make_creation_wrapper(fn: Callable[..., Any], name: str) -> Callable[..., Any]:
    """Wrap a tensor creation function to intercept dtype."""
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if "dtype" in kwargs:
            kwargs["dtype"] = _resolve_dtype(kwargs["dtype"])
        return fn(*args, **kwargs)
    return wrapper


_CREATION_FN_NAMES = ["tensor", "zeros", "ones", "randn", "empty", "full", "rand", "zeros_like",
                      "ones_like", "randn_like", "empty_like"]


def _patch_creation_fns() -> None:
    """Patch tensor creation functions."""
    global _patched  # noqa: PLW0603
    try:
        import torch
    except ImportError:
        return

    for name in _CREATION_FN_NAMES:
        fn = getattr(torch, name, None)
        if fn is not None and name not in _originals_creation:
            _originals_creation[name] = fn
            setattr(torch, name, _make_creation_wrapper(fn, name))

    _patched = True
    logger.debug("Dtype management patches applied to %d creation functions", len(_originals_creation))


def _unpatch_creation_fns() -> None:
    """Restore original tensor creation functions."""
    global _patched  # noqa: PLW0603
    try:
        import torch
    except ImportError:
        return

    for name, fn in _originals_creation.items():
        setattr(torch, name, fn)

    _originals_creation.clear()
    _patched = False
    logger.debug("Dtype management patches removed")
