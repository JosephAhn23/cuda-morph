"""Public exception hierarchy for cuda-morph.

All cuda-morph exceptions inherit from :class:`CudaMorphError` (alias for
:class:`AscendCompatError`), so users can ``except CudaMorphError`` to catch
any library error, or be specific with a subclass.

Example::

    from ascend_compat.exceptions import BackendNotFoundError, CudaMorphError

    try:
        ascend_compat.activate()
    except BackendNotFoundError:
        print("No accelerator found — falling back to CPU")
    except CudaMorphError as e:
        print(f"cuda-morph error: {e}")
"""

from ascend_compat._exceptions import (  # noqa: F401
    AscendCompatError,
    ActivationError,
    BackendNotFoundError,
    CompatibilityError,
    CudaMorphError,
    PatchError,
    PortError,
    SecurityError,
    ValidationError,
)

__all__ = [
    "AscendCompatError",
    "CudaMorphError",
    "ActivationError",
    "BackendNotFoundError",
    "CompatibilityError",
    "PatchError",
    "PortError",
    "SecurityError",
    "ValidationError",
]
