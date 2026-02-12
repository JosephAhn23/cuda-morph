"""ascend-compat: CUDA → Ascend NPU compatibility shim for PyTorch.

This is **not** a replacement for torch_npu.  torch_npu already handles the
hard C++/CANN integration via PyTorch's PrivateUse1 dispatch mechanism.
ascend-compat is a *thin, high-value ecosystem compatibility bridge* that
fixes the last mile: existing CUDA-assuming Python code that hard-codes
``torch.cuda`` calls.

Architecture (four-layer stack)::

    ┌─────────────────────────────────────────────────────┐
    │  Layer 4: ascend_compat.doctor                      │
    │  Environment validation, error translation,         │
    │  diagnostics CLI                                    │
    ├─────────────────────────────────────────────────────┤
    │  Layer 3: ascend_compat.ecosystem                   │
    │  HuggingFace, DeepSpeed, flash-attn, vLLM shims    │
    ├─────────────────────────────────────────────────────┤
    │  Layer 2: ascend_compat.cuda_shim                   │
    │  torch.cuda API interception + intelligent routing  │
    ├─────────────────────────────────────────────────────┤
    │  Layer 1: torch_npu (Huawei — already exists)       │
    │  PrivateUse1 backend, C++ dispatch, CANN/ACL        │
    └─────────────────────────────────────────────────────┘

Activation Modes
----------------
``import ascend_compat`` does **not** automatically patch ``torch.cuda``
by default.  This is a deliberate design choice: imports should not have
global side effects, especially when ascend-compat might be imported
transitively by a library.

There are three ways to activate the shim:

1. **Explicit activation** (recommended for applications)::

       import ascend_compat
       ascend_compat.activate()

2. **CLI launcher** (recommended for running existing scripts unchanged)::

       ascend-compat run script.py

3. **Environment variable** (opt-in to auto-activate on import)::

       export ASCEND_COMPAT_AUTO_ACTIVATE=1
       python script.py  # import ascend_compat now auto-activates

To prevent activation entirely (e.g. in testing)::

    export ASCEND_COMPAT_NO_PATCH=1

After activation the shim:
- Detects your hardware (NPU > CUDA > CPU)
- Routes ``torch.cuda.*`` calls to ``torch.npu.*`` equivalents
- Makes ``torch.cuda.is_available()`` return ``False`` to prevent the
  NCCL-vs-HCCL misdetection bug in accelerate/DeepSpeed
- Patches ``torch.device("cuda")`` → ``torch.device("npu")``
- Patches ``Tensor.cuda()`` → ``Tensor.npu()``

For ecosystem-specific fixes::

    from ascend_compat.ecosystem import transformers_patch
    transformers_patch.apply()  # Fixes device_map="auto" on NPU

    from ascend_compat.ecosystem import flash_attn  # Drop-in flash_attn replacement

Observability
-------------
After activation, you can inspect which patches are being hit::

    stats = ascend_compat.get_patch_stats()
    # => {"cuda.is_available": 42, "torch.device": 137, ...}

Migration from v0.2.x
---------------------
In v0.2.x, ``import ascend_compat`` auto-activated the shim.  As of v0.3.0+,
you must explicitly call ``ascend_compat.activate()`` or use the CLI launcher.
See MIGRATION.md for details.

Environment Variables
---------------------
``ASCEND_COMPAT_AUTO_ACTIVATE``
    Set to ``1`` to auto-activate on ``import ascend_compat``.
``ASCEND_COMPAT_LOG_LEVEL``
    Set to ``DEBUG`` to see every API translation. Default: ``WARNING``.
``ASCEND_COMPAT_NO_PATCH``
    Set to ``1`` to prevent activation entirely (even explicit calls).
"""

from __future__ import annotations

import os
import warnings

__version__ = "0.8.0"

# Core infrastructure (always available — no side effects)
from ascend_compat._backend import (
    Backend,
    detect_backends,
    has_cuda,
    has_mlu,
    has_npu,
    preferred_backend,
)
from ascend_compat._logging import set_log_level

# Layer 2: CUDA shim (activation is explicit, not on import)
from ascend_compat.cuda_shim import (
    activate,
    deactivate,
    get_all_patch_stats,
    get_patch_stats,
    is_activated,
    reset_patch_stats,
)

# ---------------------------------------------------------------------------
# Backward compatibility: deprecation warning for v0.2.x users
# ---------------------------------------------------------------------------
# In v0.2.x, `import ascend_compat` auto-activated the shim.  We removed that
# in v0.3.0+ because library imports shouldn't have global side effects.
# Emit a one-time deprecation warning if the user appears to be relying on
# the old behavior (i.e. they imported us but haven't called activate()).

import atexit as _atexit
import sys as _sys


def _check_activation_at_exit() -> None:
    """Emit a deprecation warning at exit if shim was never activated.

    This catches the v0.2.x pattern where users relied on import-time
    activation.  We only warn if:
    1. The shim was imported but never activated
    2. We're not in a test runner (pytest sets 'pytest' in sys.modules)
    3. We haven't already warned
    """
    if is_activated():
        return
    if "pytest" in _sys.modules:
        return  # Don't warn during testing
    if os.environ.get("ASCEND_COMPAT_NO_PATCH", "").strip() == "1":
        return  # User explicitly disabled patches

    # Check if ascend_compat was imported in user code (not just transitively)
    import_in_main = False
    main_mod = _sys.modules.get("__main__")
    if main_mod is not None:
        src = getattr(main_mod, "__file__", "") or ""
        if src:
            try:
                with open(src) as f:
                    content = f.read()
                import_in_main = "ascend_compat" in content
            except (OSError, IOError):
                pass

    if import_in_main:
        warnings.warn(
            "ascend-compat was imported but activate() was never called. "
            "Since v0.3.0, auto-activation on import is removed. "
            "Add `ascend_compat.activate()` after import, use "
            "`ascend-compat run script.py`, or set "
            "ASCEND_COMPAT_AUTO_ACTIVATE=1. "
            "See MIGRATION.md for details.",
            DeprecationWarning,
            stacklevel=1,
        )


_atexit.register(_check_activation_at_exit)

# ---------------------------------------------------------------------------
# Conditional auto-activation
# ---------------------------------------------------------------------------
# We only auto-activate when the user has explicitly opted in via env var.
# This prevents the "library import has global side effects" problem.
# The CLI launcher (ascend-compat run) sets this var automatically.
if os.environ.get("ASCEND_COMPAT_AUTO_ACTIVATE", "").strip() == "1":
    activate()

__all__ = [
    # Version
    "__version__",
    # Backend introspection
    "Backend",
    "detect_backends",
    "preferred_backend",
    "has_npu",
    "has_mlu",
    "has_cuda",
    # Shim control
    "activate",
    "deactivate",
    "is_activated",
    "set_log_level",
    # Telemetry
    "get_patch_stats",
    "get_all_patch_stats",
    "reset_patch_stats",
]
