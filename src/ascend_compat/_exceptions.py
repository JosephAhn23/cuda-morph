"""Custom exception hierarchy for ascend-compat.

Provides distinct exception types so consumers can distinguish
ascend-compat errors from unrelated failures in their except clauses.

Usage::

    from ascend_compat._exceptions import ActivationError

    try:
        ascend_compat.activate()
    except ActivationError as e:
        print(f"Shim activation failed: {e}")
"""

from __future__ import annotations


class AscendCompatError(Exception):
    """Base exception for all ascend-compat errors.

    Catch this to handle any error raised by the library without
    catching unrelated exceptions.
    """


class ActivationError(AscendCompatError):
    """Raised when shim activation fails (e.g. patch application error).

    The shim guarantees atomic rollback — if this is raised, no patches
    were left in a half-applied state.
    """


class BackendNotFoundError(AscendCompatError):
    """Raised when a required backend or adapter is not available.

    Examples:
    - ``torch_npu`` is not installed but NPU operations are requested
    - ``npu_fusion_attention`` is missing from torch_npu
    """


class PatchError(AscendCompatError):
    """Raised when an individual patch cannot be applied or reverted."""


class CompatibilityError(AscendCompatError):
    """Raised when a version or compatibility check fails hard.

    Soft failures emit warnings; this is for fatal incompatibilities
    (e.g. known-bad torch_npu + PyTorch combinations).
    """


class PortError(AscendCompatError):
    """Raised when code porting/rewriting fails."""


class ValidationError(AscendCompatError):
    """Raised when operator verification fails."""


class SecurityError(AscendCompatError):
    """Raised when a security or integrity check fails."""


# Keep the public alias for backward compat
CudaMorphError = AscendCompatError
