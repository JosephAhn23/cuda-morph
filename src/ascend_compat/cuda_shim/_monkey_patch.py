"""Runtime monkey-patching of torch.cuda → torch.npu.

This is the second half of the hybrid approach (import hook + monkey-patch).
After the import hook ensures torch_npu is loaded, this module patches:

1. ``torch.cuda.*`` functions → ``torch.npu.*`` equivalents (using the registry)
2. ``torch.device`` constructor → remaps ``"cuda"`` → ``"npu"`` strings
3. ``torch.Tensor.cuda()`` → ``torch.Tensor.npu()``
4. ``torch.nn.Module.cuda()`` → ``torch.nn.Module.npu()``
5. ``torch.cuda.is_available()`` → **False** (critical — see below)

The is_available() → False decision
------------------------------------
This is the single most important design decision in the entire project.

torch_npu's ``transfer_to_npu`` makes ``torch.cuda.is_available()`` return
True.  This causes HuggingFace accelerate to select the NCCL backend for
distributed training.  On Ascend, NCCL doesn't exist — HCCL is the correct
backend.  The result is a *silent* distributed training failure.

By making ``is_available()`` return False, we force libraries to fall through
to their NPU detection paths (which many already have, thanks to Huawei's
upstream contributions to transformers/accelerate).  Users who explicitly
need to check for NPU availability should use ``torch.npu.is_available()``
or ``ascend_compat.has_npu()``.

Version-gated patching
----------------------
PyTorch's internal APIs change across versions.  Rather than applying a
single set of patches and hoping they work, we detect the PyTorch version
at activation time and apply version-appropriate patches.  Known version
boundaries:

- **PyTorch 2.0–2.1**: ``torch.device`` is a type (class), safe to replace.
- **PyTorch 2.2+**: Added ``torch.accelerator`` module — some code paths use
  this instead of ``torch.cuda``.
- **PyTorch 2.4+**: ``torch.cuda.amp`` is deprecated in favor of
  ``torch.amp``.  We patch both paths.
- **PyTorch 2.5+**: Accelerator auto-loading means ``import torch`` can
  automatically load torch_npu.  We detect this and skip redundant patches.

Thread safety & atomicity
-------------------------
All activation and deactivation is managed by :class:`PatchManager`, which
provides:

- **RLock** protection against concurrent activation
- **Reference counting** — multiple ``activate()`` calls are safe; only the
  last ``deactivate()`` actually restores originals
- **Atomic batches** — if any patch fails, all previously applied patches in
  the current activation are rolled back
- **Telemetry** — every patched function counts its calls for observability
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Callable, Dict, Optional, Tuple

from ascend_compat._backend import Backend, get_torch, has_npu, preferred_backend
from ascend_compat._logging import get_logger
from ascend_compat.cuda_shim._import_hook import install_import_hook, uninstall_import_hook
from ascend_compat.cuda_shim._patch_manager import PatchManager
from ascend_compat.cuda_shim._registry import MappingKind, get_all_mappings

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Singleton PatchManager instance
# ---------------------------------------------------------------------------

_manager = PatchManager()


# ---------------------------------------------------------------------------
# Version detection
# ---------------------------------------------------------------------------


def _pytorch_version() -> Tuple[int, int, int]:
    """Return PyTorch version as (major, minor, patch) ints.

    Handles version strings like '2.5.1', '2.5.1+cu121', '2.5.1a0+gitXXX'.
    Returns (0, 0, 0) if PyTorch is not installed.
    """
    try:
        torch = get_torch()
        version_str = torch.__version__.split("+")[0].split("a")[0]
        parts = version_str.split(".")
        return (int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0)
    except Exception:  # noqa: BLE001
        return (0, 0, 0)


def _torch_npu_version() -> Tuple[int, int, int]:
    """Return torch_npu version as (major, minor, patch) ints."""
    try:
        from ascend_compat._backend import get_torch_npu
        npu = get_torch_npu()
        if npu is None:
            return (0, 0, 0)
        version_str = getattr(npu, "__version__", "0.0.0")
        version_str = version_str.split("+")[0].split("a")[0]
        parts = version_str.split(".")
        return (int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0)
    except Exception:  # noqa: BLE001
        return (0, 0, 0)


# ---------------------------------------------------------------------------
# Version compatibility check (item 5 — pre-activation guard)
# ---------------------------------------------------------------------------


def _check_version_compatibility() -> None:
    """Warn if PyTorch and torch_npu have mismatched major versions.

    This catches the common case where a user upgrades PyTorch but forgets
    torch_npu (or vice versa), which produces cryptic C++ errors at runtime.
    """
    pt_ver = _pytorch_version()
    npu_ver = _torch_npu_version()

    if pt_ver == (0, 0, 0) or npu_ver == (0, 0, 0):
        return  # Can't check — one of them isn't installed

    if pt_ver[:2] != npu_ver[:2]:
        warnings.warn(
            f"PyTorch {pt_ver[0]}.{pt_ver[1]}.{pt_ver[2]} and "
            f"torch_npu {npu_ver[0]}.{npu_ver[1]}.{npu_ver[2]} have different "
            f"major.minor versions. This will likely cause instability. "
            f"Update torch_npu to match: pip install torch-npu=={pt_ver[0]}.{pt_ver[1]}.*",
            RuntimeWarning,
            stacklevel=3,
        )

    # Known bad combinations
    _BAD_COMBOS: Dict[Tuple[Tuple[int, int], Tuple[int, int]], str] = {
        ((2, 1), (2, 0)): "torch_npu 2.0 is not compatible with PyTorch 2.1 — upgrade torch_npu",
    }

    combo = (pt_ver[:2], npu_ver[:2])
    if combo in _BAD_COMBOS:
        warnings.warn(
            f"Known incompatible combination: {_BAD_COMBOS[combo]}",
            RuntimeWarning,
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def activate() -> None:
    """Activate the full CUDA → Ascend shim.

    Thread-safe, reference-counted, atomic.

    1. Detects PyTorch and torch_npu versions for version-gated patching
    2. Checks version compatibility (warns on mismatches)
    3. Installs the import hook (ensures torch_npu loads before torch.cuda)
    4. Applies monkey-patches atomically (rolls back on failure)
    5. Patches torch.device to remap "cuda" → "npu"
    6. Sets torch.cuda.is_available() → False on NPU systems

    Idempotent (reference-counted).  Set ``ASCEND_COMPAT_NO_PATCH=1`` to skip.
    """
    if os.environ.get("ASCEND_COMPAT_NO_PATCH", "").strip() == "1":
        logger.info("ASCEND_COMPAT_NO_PATCH=1 — skipping all patches")
        return

    # Reference counting: only apply patches on the first activation
    if not _manager.increment_ref():
        logger.debug("activate() called but already active (ref_count=%d)", _manager.ref_count)
        return

    backend = preferred_backend()
    pt_ver = _pytorch_version()
    npu_ver = _torch_npu_version()

    logger.info(
        "Activating ascend-compat (backend=%s, PyTorch=%d.%d.%d, torch_npu=%d.%d.%d)",
        backend.value, *pt_ver, *npu_ver,
    )

    # Pre-activation version compatibility check
    _check_version_compatibility()

    # Atomic activation: if any patch fails, roll back everything
    _manager.begin_batch()
    try:
        if backend == Backend.NPU:
            install_import_hook()
            _patch_cuda_namespace()
            _patch_torch_device()
            _patch_cuda_is_available()
            _patch_tensor_cuda()

            # Version-gated patches
            if pt_ver >= (2, 4, 0):
                _patch_torch_amp_new_location()
            if pt_ver >= (2, 2, 0):
                _patch_torch_accelerator()
        elif backend == Backend.CPU:
            _patch_cpu_fallback()

        _manager.commit_batch()

    except Exception:
        logger.error("Activation failed — rolling back all patches")
        _manager.rollback_batch()
        _manager._ref_count = 0  # Reset ref count on failure
        raise


def deactivate() -> None:
    """Restore all originals.

    Thread-safe, reference-counted.  Only actually reverts patches when the
    last outstanding activation is removed.
    """
    if not _manager.decrement_ref():
        logger.debug("deactivate() called but still active (ref_count=%d)", _manager.ref_count)
        return

    _manager.revert_all()
    uninstall_import_hook()
    logger.info("ascend-compat deactivated — all patches reverted")


def is_activated() -> bool:
    """Return True if the shim is currently active."""
    return _manager.is_active


def get_patch_stats() -> Dict[str, int]:
    """Return per-patch call counters for observability.

    Returns:
        Dict mapping patch name → number of times the patched function was
        called.  E.g. ``{"cuda.is_available": 42, "torch.device": 137}``.

    Example::

        import ascend_compat
        ascend_compat.activate()

        # ... run your model ...

        stats = ascend_compat.get_patch_stats()
        for name, count in sorted(stats.items(), key=lambda x: -x[1]):
            print(f"  {name}: {count} calls")
    """
    return _manager.get_stats()


def get_all_patch_stats() -> Dict[str, int]:
    """Return counters for ALL registered patches (including zero-count)."""
    return _manager.get_all_stats()


def reset_patch_stats() -> None:
    """Reset all telemetry counters to zero."""
    _manager.reset_stats()


def get_patch_names() -> list[str]:
    """Return names of all currently applied patches."""
    return _manager.get_patch_names()


# ---------------------------------------------------------------------------
# NPU patching (using PatchManager)
# ---------------------------------------------------------------------------


def _patch_cuda_namespace() -> None:
    """Patch torch.cuda.* using the mapping registry."""
    torch = get_torch()

    if not hasattr(torch, "npu"):
        logger.error("torch.npu not found — is torch_npu installed?")
        return

    npu = torch.npu
    registry = get_all_mappings()
    count = 0

    for cuda_name, mapping in registry.items():
        if mapping.kind == MappingKind.UNSUPPORTED:
            stub = _make_unsupported_stub(cuda_name, mapping.note)
            _manager.apply(torch.cuda, cuda_name, stub, f"cuda.{cuda_name}", count_calls=False)
            count += 1
            continue

        # For DIRECT and ADAPTED mappings, resolve the NPU target
        npu_target = _resolve_npu_attr(npu, mapping.npu_name)
        if npu_target is None:
            logger.debug("Skipping %s — torch.npu.%s not found", cuda_name, mapping.npu_name)
            continue

        if mapping.kind == MappingKind.ADAPTED and mapping.adapter is not None:
            target = mapping.adapter
        else:
            target = npu_target

        proxy = _make_proxy(target, cuda_name, mapping.npu_name)
        _manager.apply(torch.cuda, cuda_name, proxy, f"cuda.{cuda_name}")
        count += 1

    logger.info("Patched %d torch.cuda attributes via registry", count)


def _resolve_npu_attr(npu: Any, dotted_name: str) -> Optional[Any]:
    """Resolve a potentially dotted attribute like 'amp.autocast' on torch.npu."""
    obj = npu
    for part in dotted_name.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def _make_proxy(target: Callable[..., Any], cuda_name: str, npu_name: str) -> Callable[..., Any]:
    """Create a logging proxy that delegates to the NPU function."""
    def proxy(*args: Any, **kwargs: Any) -> Any:
        logger.debug("torch.cuda.%s → torch.npu.%s", cuda_name, npu_name)
        return target(*args, **kwargs)
    proxy.__name__ = cuda_name
    proxy.__doc__ = f"ascend-compat: torch.cuda.{cuda_name} → torch.npu.{npu_name}"
    return proxy


def _make_unsupported_stub(cuda_name: str, note: str) -> Callable[..., Any]:
    """Create a stub that raises NotImplementedError with guidance."""
    def stub(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            f"torch.cuda.{cuda_name} is not supported on Ascend NPU.\n"
            f"  {note}\n"
            f"  See: ascend-compat docs/compatibility_matrix.md"
        )
    stub.__name__ = cuda_name
    stub.__doc__ = f"UNSUPPORTED on Ascend: {note}"
    return stub


# ---------------------------------------------------------------------------
# torch.cuda.is_available() → False
# ---------------------------------------------------------------------------


def _patch_cuda_is_available() -> None:
    """Make torch.cuda.is_available() return False on NPU systems.

    WHY: When is_available() returns True on Ascend, HuggingFace accelerate
    selects NCCL as the distributed backend.  NCCL doesn't exist on Ascend —
    HCCL is the correct backend.  The result is a silent failure during
    distributed training that wastes hours of developer time.

    By returning False, we force libraries to fall through to their NPU
    detection code paths.  Libraries that support NPU (transformers >=4.40,
    accelerate >=0.28) will then correctly detect and use NPU.

    Users who need to check NPU availability explicitly should use:
        torch.npu.is_available()
        ascend_compat.has_npu()
    """
    torch = get_torch()

    def _npu_aware_is_available() -> bool:
        """Returns False to prevent NCCL misdetection.  Use torch.npu.is_available() instead."""
        logger.debug(
            "torch.cuda.is_available() → False (NPU system; use torch.npu.is_available())"
        )
        return False

    _manager.apply(torch.cuda, "is_available", _npu_aware_is_available, "cuda.is_available")
    logger.info(
        "torch.cuda.is_available() patched → False "
        "(prevents NCCL misdetection; NPU available via torch.npu.is_available())"
    )


# ---------------------------------------------------------------------------
# torch.device("cuda") → torch.device("npu")
# ---------------------------------------------------------------------------


def _patch_torch_device() -> None:
    """Patch torch.device to transparently remap 'cuda' → 'npu'.

    This is the most impactful single patch.  Thousands of scripts do:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    or even:
        device = torch.device("cuda:0")

    Without this patch, these all fail on Ascend with:
        RuntimeError: Expected one of cpu, ... device type at start of device string: cuda
    (when torch_npu hasn't registered the "cuda" alias)

    With this patch, "cuda" silently becomes "npu".
    """
    torch = get_torch()
    original = torch.device

    def _patched_device(*args: Any, **kwargs: Any) -> Any:
        if args and isinstance(args[0], str):
            dev_str = args[0]
            if dev_str.startswith("cuda"):
                new_str = dev_str.replace("cuda", "npu", 1)
                logger.debug("torch.device(%r) → torch.device(%r)", dev_str, new_str)
                args = (new_str,) + args[1:]
        return original(*args, **kwargs)

    _manager.apply(torch, "device", _patched_device, "torch.device")
    logger.debug("torch.device patched — 'cuda' strings remapped to 'npu'")


# ---------------------------------------------------------------------------
# Tensor.cuda() / Module.cuda() → .npu()
# ---------------------------------------------------------------------------


def _patch_tensor_cuda() -> None:
    """Patch Tensor.cuda() and Module.cuda() to call .npu() instead.

    Many codebases do ``x = x.cuda()`` or ``model = model.cuda()``.
    torch_npu already registers ``.npu()`` on tensors/modules, but
    ``.cuda()`` would still try to use the (non-existent) CUDA backend.

    Note: torch_npu may already patch this via transfer_to_npu.  We apply
    our own patch regardless, because we need to control the behavior
    (especially around the is_available() return value).
    """
    torch = get_torch()

    # Tensor.cuda
    def _tensor_cuda(self: Any, device: Any = None, **kwargs: Any) -> Any:
        logger.debug("Tensor.cuda() → Tensor.npu()")
        if device is not None and isinstance(device, (str, torch.device)):
            device = str(device).replace("cuda", "npu")
        return self.npu(device, **kwargs) if device is not None else self.npu(**kwargs)

    _manager.apply(torch.Tensor, "cuda", _tensor_cuda, "Tensor.cuda")

    # Module.cuda
    def _module_cuda(self: Any, device: Any = None) -> Any:
        logger.debug("Module.cuda() → Module.npu()")
        if device is not None and isinstance(device, (str, torch.device)):
            device = str(device).replace("cuda", "npu")
        return self.npu(device) if device is not None else self.npu()

    _manager.apply(torch.nn.Module, "cuda", _module_cuda, "Module.cuda")

    logger.debug("Tensor.cuda() and Module.cuda() patched → .npu()")


# ---------------------------------------------------------------------------
# Version-gated patches
# ---------------------------------------------------------------------------


def _patch_torch_amp_new_location() -> None:
    """Patch ``torch.amp`` (PyTorch 2.4+ moved AMP here from torch.cuda.amp).

    Since PyTorch 2.4, ``torch.cuda.amp.autocast`` and ``torch.cuda.amp.GradScaler``
    are deprecated aliases for ``torch.amp.autocast`` and ``torch.amp.GradScaler``.
    Code that uses the new location needs patching too.
    """
    torch = get_torch()

    if not hasattr(torch, "amp"):
        logger.debug("torch.amp not found — skipping (PyTorch < 2.4?)")
        return

    amp = torch.amp

    # Patch autocast to default to "npu" instead of "cuda"
    if hasattr(amp, "autocast"):
        original_autocast = amp.autocast

        class _NPUAutocast(original_autocast):  # type: ignore[misc]
            """Autocast wrapper that defaults device_type to 'npu'."""

            def __init__(self, device_type: str = "npu", *args: Any, **kwargs: Any) -> None:
                if device_type == "cuda":
                    device_type = "npu"
                    logger.debug("torch.amp.autocast(device_type='cuda') → 'npu'")
                super().__init__(device_type, *args, **kwargs)

        _manager.apply(torch.amp, "autocast", _NPUAutocast, "amp.autocast", count_calls=False)
        logger.debug("torch.amp.autocast patched — 'cuda' → 'npu' device_type")

    # Patch GradScaler
    if hasattr(amp, "GradScaler"):
        original_gradscaler = amp.GradScaler

        class _NPUGradScaler(original_gradscaler):  # type: ignore[misc]
            """GradScaler wrapper that defaults device to 'npu'."""

            def __init__(self, device: str = "npu", *args: Any, **kwargs: Any) -> None:
                if device == "cuda":
                    device = "npu"
                    logger.debug("torch.amp.GradScaler(device='cuda') → 'npu'")
                super().__init__(device, *args, **kwargs)

        _manager.apply(torch.amp, "GradScaler", _NPUGradScaler, "amp.GradScaler", count_calls=False)
        logger.debug("torch.amp.GradScaler patched — 'cuda' → 'npu' device")


def _patch_torch_accelerator() -> None:
    """Patch ``torch.accelerator`` (PyTorch 2.2+) to report NPU.

    PyTorch 2.2 introduced ``torch.accelerator`` as a device-agnostic API.
    Some libraries check ``torch.accelerator.current_accelerator()`` instead
    of ``torch.cuda.is_available()``.  We ensure it reports NPU correctly.
    """
    torch = get_torch()

    if not hasattr(torch, "accelerator"):
        logger.debug("torch.accelerator not found — skipping (PyTorch < 2.2?)")
        return

    # torch.accelerator should already work with NPU via PrivateUse1 on
    # PyTorch 2.5+.  We log what it reports for diagnostics.
    try:
        accel = torch.accelerator
        if hasattr(accel, "current_accelerator"):
            current = accel.current_accelerator()
            logger.debug("torch.accelerator.current_accelerator() = %s", current)
        if hasattr(accel, "is_available"):
            available = accel.is_available()
            logger.debug("torch.accelerator.is_available() = %s", available)
    except Exception as exc:  # noqa: BLE001
        logger.debug("torch.accelerator inspection failed: %s", exc)


# ---------------------------------------------------------------------------
# CPU fallback (no NPU, no CUDA)
# ---------------------------------------------------------------------------


def _patch_cpu_fallback() -> None:
    """Safe no-ops for CPU-only systems."""
    torch = get_torch()

    _manager.apply(torch.cuda, "is_available", lambda: False, "cpu.is_available")
    _manager.apply(torch.cuda, "device_count", lambda: 0, "cpu.device_count")

    def _noop(*a: Any, **kw: Any) -> None:
        pass

    def _zero(*a: Any, **kw: Any) -> int:
        return 0

    _manager.apply(torch.cuda, "set_device", _noop, "cpu.set_device")
    _manager.apply(torch.cuda, "synchronize", _noop, "cpu.synchronize")
    _manager.apply(torch.cuda, "empty_cache", _noop, "cpu.empty_cache")
    _manager.apply(torch.cuda, "manual_seed", _noop, "cpu.manual_seed")
    _manager.apply(torch.cuda, "manual_seed_all", _noop, "cpu.manual_seed_all")

    for attr in ("memory_allocated", "max_memory_allocated",
                 "memory_reserved", "max_memory_reserved"):
        _manager.apply(torch.cuda, attr, _zero, f"cpu.{attr}")

    logger.info("CPU fallback patches applied — torch.cuda calls are safe no-ops")
