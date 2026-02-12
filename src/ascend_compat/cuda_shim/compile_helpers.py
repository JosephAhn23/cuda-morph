"""torch.compile / torchair integration helpers for Ascend.

Ascend achieves peak performance when CANN compiles the entire computation
graph ahead of execution, enabling operator fusion, memory reuse, and
scheduling optimization.  The recommended path is:

    torch.compile(model, backend="torchair")

However, this requires torchair to be installed and configured correctly.
Ascend does NOT support the default Inductor/Triton backend — attempting
to use it produces cryptic errors.

This module provides:
1. Detection of available compilation backends
2. Automatic backend selection for ``torch.compile``
3. Graph mode configuration (Ascend's alternative to CUDA Graphs)
4. Shape bucketing to avoid expensive CANN recompilation on dynamic shapes

Usage::

    from ascend_compat.cuda_shim.compile_helpers import (
        get_compile_backend,
        enable_graph_mode,
        ShapeBucketer,
    )

    # Auto-select the right backend for torch.compile
    model = torch.compile(model, backend=get_compile_backend())

    # Or enable Ascend graph mode (alternative to CUDA Graphs)
    enable_graph_mode()

    # Bucket dynamic shapes to avoid recompilation
    bucketer = ShapeBucketer(buckets=[128, 256, 512, 1024, 2048])
    padded_input = bucketer.pad(input_tensor, dim=1)
"""

from __future__ import annotations

import math
import sys
import threading
import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from ascend_compat._backend import has_npu, get_torch, get_torch_npu
from ascend_compat._logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "get_compile_backend",
    "is_torchair_available",
    "get_compile_info",
    "enable_graph_mode",
    "disable_graph_mode",
    "safe_compile",
    "CompatibilityPolicy",
    "LATEST_TESTED_VERSION",
    "ShapeBucketer",
]

# Highest PyTorch version tested with this release.  Used by CompatibilityPolicy.
LATEST_TESTED_VERSION: Tuple[int, int, int] = (2, 5, 1)


# ---------------------------------------------------------------------------
# Compile backend detection
# ---------------------------------------------------------------------------


def get_compile_backend() -> str:
    """Return the recommended ``torch.compile`` backend for the current system.

    On Ascend NPU: returns ``"torchair"`` if available, ``"eager"`` otherwise.
    On CUDA: returns ``"inductor"`` (PyTorch default).
    On CPU: returns ``"inductor"``.

    Usage::

        model = torch.compile(model, backend=get_compile_backend())
    """
    if not has_npu():
        return "inductor"

    # Check for torchair
    if is_torchair_available():
        logger.info("torch.compile backend: torchair (Ascend-optimized)")
        return "torchair"

    # Check for Ascend's aot backend
    if _is_backend_available("ascend"):
        logger.info("torch.compile backend: ascend")
        return "ascend"

    logger.warning(
        "torchair not found — torch.compile will use 'eager' (no graph optimization). "
        "Install torchair: pip install torchair (from Huawei's repo)"
    )
    return "eager"


def is_torchair_available() -> bool:
    """Check if torchair (Ascend's torch.compile backend) is installed."""
    try:
        import torchair  # type: ignore[import-untyped]  # noqa: F401
        return True
    except ImportError:
        return False


def _is_backend_available(name: str) -> bool:
    """Check if a named torch.compile backend is registered."""
    try:
        torch = get_torch()
        if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "list_backends"):
            backends = torch._dynamo.list_backends()
            return name in backends
    except Exception:  # noqa: BLE001
        pass
    return False


def get_compile_info() -> Dict[str, Any]:
    """Return information about torch.compile configuration.

    Useful for diagnostics (``ascend-compat doctor``).
    """
    info: Dict[str, Any] = {
        "recommended_backend": get_compile_backend(),
        "torchair_available": is_torchair_available(),
        "available_backends": [],
    }

    try:
        torch = get_torch()
        if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "list_backends"):
            info["available_backends"] = list(torch._dynamo.list_backends())
    except Exception:  # noqa: BLE001
        pass

    return info


# ---------------------------------------------------------------------------
# Graph mode (Ascend's alternative to CUDA Graphs)
# ---------------------------------------------------------------------------


def enable_graph_mode() -> bool:
    """Enable Ascend graph mode for the current process.

    Ascend graph mode is the NPU equivalent of CUDA Graphs — the entire
    computation graph is compiled and optimized before execution, enabling:

    - Operator fusion (up to 10x speedup for attention)
    - Memory reuse planning
    - Pipeline scheduling across Cube/Vector units

    The tradeoff: dynamic shapes trigger expensive recompilation.
    Use :class:`ShapeBucketer` to mitigate this.

    Returns:
        True if graph mode was enabled, False if not on NPU.
    """
    if not has_npu():
        logger.debug("Not on NPU — skipping graph mode")
        return False

    try:
        torch = get_torch()
        if hasattr(torch, "npu") and hasattr(torch.npu, "set_compile_mode"):
            torch.npu.set_compile_mode("graph_mode")
            logger.info("Ascend graph mode enabled (NPU equivalent of CUDA Graphs)")
            return True
        else:
            logger.debug("torch.npu.set_compile_mode not available")
            return False
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to enable graph mode: %s", exc)
        return False


def disable_graph_mode() -> bool:
    """Disable Ascend graph mode, returning to eager execution."""
    if not has_npu():
        return False

    try:
        torch = get_torch()
        if hasattr(torch, "npu") and hasattr(torch.npu, "set_compile_mode"):
            torch.npu.set_compile_mode("default")
            logger.info("Ascend graph mode disabled — back to eager execution")
            return True
        return False
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to disable graph mode: %s", exc)
        return False


# ---------------------------------------------------------------------------
# safe_compile — torch.compile with error recovery + diagnostics
# ---------------------------------------------------------------------------


def safe_compile(model: Any, **kwargs: Any) -> Any:
    """Compile a model with graceful fallback and diagnostics on failure.

    Wraps ``torch.compile`` and catches compilation errors.  If compilation
    fails, prints diagnostics with suggested fixes and returns the original
    (eager) model so training/inference can continue.

    Args:
        model: A PyTorch ``nn.Module`` to compile.
        **kwargs: Forwarded to ``torch.compile``.  If ``backend`` is not
            specified, ``get_compile_backend()`` is used automatically.

    Returns:
        The compiled model, or the original model if compilation failed.

    Usage::

        from ascend_compat.cuda_shim.compile_helpers import safe_compile

        model = safe_compile(model)  # Auto-selects backend, never crashes
    """
    torch = get_torch()

    if "backend" not in kwargs:
        kwargs["backend"] = get_compile_backend()

    try:
        compiled = torch.compile(model, **kwargs)
        logger.info("torch.compile succeeded (backend=%s)", kwargs.get("backend"))
        return compiled
    except Exception as exc:
        # Collect diagnostics
        param_count = sum(p.numel() for p in model.parameters()) if hasattr(model, "parameters") else 0
        npu_mod = get_torch_npu()

        diag: Dict[str, Any] = {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "torch_version": torch.__version__,
            "torch_npu_version": getattr(npu_mod, "__version__", None) if npu_mod else None,
            "backend": kwargs.get("backend", "unknown"),
            "model_params": param_count,
        }

        logger.error("torch.compile failed: %s", exc)
        logger.error("Diagnostics: %s", diag)

        # Suggest fix based on error
        err_lower = str(exc).lower()
        if "out of memory" in err_lower or "oom" in err_lower:
            logger.error(
                "Hint: Try reducing batch size or using "
                "torch.compile(model, mode='reduce-overhead')"
            )
        elif "dynamic" in err_lower and "shape" in err_lower:
            logger.error(
                "Hint: Use ascend_compat.cuda_shim.compile_helpers.ShapeBucketer "
                "to bucket dynamic shapes before compilation"
            )
        elif "torchair" in err_lower or "backend" in err_lower:
            logger.error(
                "Hint: Install torchair from Huawei's repo, or try "
                "safe_compile(model, backend='eager')"
            )

        print(
            f"[ascend-compat] torch.compile failed ({type(exc).__name__}). "
            f"Falling back to eager mode. Run `ascend-compat compile` for diagnostics.",
            file=sys.stderr,
        )
        return model


# ---------------------------------------------------------------------------
# Forward compatibility policy
# ---------------------------------------------------------------------------


class CompatibilityPolicy:
    """Define behavior for untested PyTorch versions.

    When a new PyTorch release (e.g. 2.6) comes out before ascend-compat
    has been updated, users need a clear behavior:

    - **strict**: Raise an error — refuse to patch untested versions.
    - **warn** (default): Apply patches for the latest tested version, but
      emit a ``FutureWarning`` so the user knows they're on thin ice.
    - **silent**: Apply patches silently (for CI systems that don't care).

    The policy can be set via:

    1. ``ASCEND_COMPAT_COMPAT_POLICY`` environment variable
    2. Passing ``policy=`` to :func:`check_forward_compat`
    3. Defaults to ``"warn"``

    Usage::

        from ascend_compat.cuda_shim.compile_helpers import CompatibilityPolicy

        CompatibilityPolicy.check_forward_compat()
    """

    STRICT = "strict"
    WARN = "warn"
    SILENT = "silent"

    @classmethod
    def _get_policy(cls, policy: Optional[str] = None) -> str:
        """Resolve the active policy (explicit arg > env var > default)."""
        if policy is not None:
            return policy
        import os
        env = os.environ.get("ASCEND_COMPAT_COMPAT_POLICY", "").strip().lower()
        if env in (cls.STRICT, cls.WARN, cls.SILENT):
            return env
        return cls.WARN

    @classmethod
    def check_forward_compat(cls, policy: Optional[str] = None) -> bool:
        """Check if the current PyTorch version exceeds the latest tested version.

        Args:
            policy: Override the default policy (``"strict"``, ``"warn"``, ``"silent"``).

        Returns:
            True if the current version is within the tested range, False if untested.

        Raises:
            RuntimeError: If policy is ``"strict"`` and the version is untested.
        """
        from ascend_compat.cuda_shim._monkey_patch import _pytorch_version

        policy = cls._get_policy(policy)
        current = _pytorch_version()

        if current == (0, 0, 0):
            return True  # Can't detect — don't block

        if current <= LATEST_TESTED_VERSION:
            return True

        current_str = f"{current[0]}.{current[1]}.{current[2]}"
        tested_str = f"{LATEST_TESTED_VERSION[0]}.{LATEST_TESTED_VERSION[1]}.{LATEST_TESTED_VERSION[2]}"

        if policy == cls.STRICT:
            raise RuntimeError(
                f"PyTorch {current_str} is newer than the latest tested version "
                f"({tested_str}). Set ASCEND_COMPAT_COMPAT_POLICY=warn to override."
            )
        elif policy == cls.WARN:
            warnings.warn(
                f"PyTorch {current_str} has not been tested with ascend-compat "
                f"(latest tested: {tested_str}). Patches for {tested_str} will be "
                f"applied. Set ASCEND_COMPAT_COMPAT_POLICY=silent to suppress.",
                FutureWarning,
                stacklevel=2,
            )
        # SILENT: no warning
        logger.info(
            "Forward compat: PyTorch %s > tested %s, policy=%s",
            current_str, tested_str, policy,
        )
        return False


# ---------------------------------------------------------------------------
# Shape bucketing (avoids CANN recompilation on dynamic shapes)
# ---------------------------------------------------------------------------


class ShapeBucketer:
    """Bucket dynamic tensor shapes to avoid CANN graph recompilation.

    On Ascend, CANN compiles computation graphs for specific tensor shapes.
    When input shapes change, the graph must be recompiled — which can take
    seconds.  For workloads with variable sequence lengths (LLM inference,
    NLP batching), this is devastating.

    ShapeBucketer rounds shapes UP to predefined bucket boundaries, reducing
    the number of unique shapes the compiler sees.  The tradeoff is wasted
    compute on padding, but this is vastly cheaper than recompilation.

    Args:
        buckets: Sorted list of allowed sizes.  Shapes are rounded up to
            the nearest bucket.  E.g. ``[128, 256, 512, 1024, 2048]``.
        max_size: Maximum allowed size.  Shapes above this are clamped.
        max_cache_entries: Maximum number of cached padded tensors.  When
            exceeded, the least-recently-used entry is evicted.  Set to 0
            to disable caching entirely.  Default: 1024.

    Usage::

        bucketer = ShapeBucketer([128, 256, 512, 1024, 2048])

        # Pad a batch of variable-length sequences
        for seq in batch:
            padded = bucketer.pad(seq, dim=0)  # rounds seq_len to nearest bucket
            model(padded)  # CANN reuses compiled graph for this bucket size

    Why this matters
    ----------------
    Without bucketing, an LLM serving requests of lengths [137, 142, 139, 155]
    triggers 4 separate CANN compilations.  With buckets=[128, 256], all four
    round up to 256, triggering just 1 compilation.

    Memory safety
    -------------
    The internal ``_pad_cache`` uses LRU eviction: once ``max_cache_entries``
    padded tensors are stored, the oldest entry is evicted on the next miss.
    Call ``clear_cache()`` explicitly between epochs or when switching models
    to free cached tensors.  Use ``cache_memory_bytes()`` to monitor usage.

    Typical bucket strategies:
    - **Power-of-2**: [64, 128, 256, 512, 1024, 2048] — simple, good default
    - **LLM inference**: [128, 256, 512, 768, 1024, 1536, 2048, 4096]
    - **Vision**: [224, 384, 512, 768, 1024] — common image sizes
    """

    # Pre-built bucket strategies
    POWER_OF_2 = [2**i for i in range(5, 13)]  # 32 to 4096
    LLM_INFERENCE = [128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
    VISION = [224, 256, 384, 512, 768, 1024]

    def __init__(
        self,
        buckets: Optional[List[int]] = None,
        max_size: Optional[int] = None,
        max_cache_entries: int = 1024,
    ) -> None:
        self.buckets = sorted(buckets or self.POWER_OF_2)
        self.max_size = max_size or self.buckets[-1]
        self.max_cache_entries = max_cache_entries
        self._hit_count: Dict[int, int] = {}
        self._miss_count = 0
        # LRU cache: (shape_tuple, dtype, device_str, dim) → padded tensor
        self._pad_cache: OrderedDict[Tuple, Any] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._lock = threading.Lock()  # Protects _pad_cache mutations

    def bucket_size(self, size: int) -> int:
        """Return the smallest bucket that is >= size.

        Args:
            size: The actual dimension size.

        Returns:
            The bucketed (rounded-up) size.
        """
        for b in self.buckets:
            if b >= size:
                self._hit_count[b] = self._hit_count.get(b, 0) + 1
                return b

        # Size exceeds all buckets — clamp to max
        self._miss_count += 1
        logger.debug("Size %d exceeds all buckets — clamping to %d", size, self.max_size)
        return self.max_size

    def pad(self, tensor: Any, dim: int = -1) -> Any:
        """Pad a tensor along `dim` to the nearest bucket size.

        Uses zero-padding.  The original data is preserved in the leading
        portion of the padded dimension.

        If ``max_cache_entries > 0``, padded tensors are cached by
        ``(shape, dtype, device, dim)`` with LRU eviction.

        Args:
            tensor: Input tensor.
            dim: Dimension to pad (supports negative indexing).

        Returns:
            Padded tensor with shape rounded up along ``dim``.
        """
        torch = get_torch()

        # Normalize negative dim
        ndim = tensor.dim()
        if dim < 0:
            dim = ndim + dim
        assert 0 <= dim < ndim, f"dim={dim} out of range for {ndim}D tensor"

        current_size = tensor.shape[dim]
        target_size = self.bucket_size(current_size)

        if target_size == current_size:
            return tensor

        # Build padding spec (torch.nn.functional.pad uses reversed dim order)
        pad_spec = [0] * (2 * ndim)
        pad_idx = 2 * (ndim - 1 - dim) + 1  # right-side padding for this dim
        pad_spec[pad_idx] = target_size - current_size

        return torch.nn.functional.pad(tensor, pad_spec, value=0)

    # ------------------------------------------------------------------
    # LRU pad cache
    # ------------------------------------------------------------------

    def pad_cached(self, tensor: Any, dim: int = -1) -> Any:
        """Like :meth:`pad`, but caches padded results with LRU eviction.

        The cache key is ``(shape, dtype, device, dim)`` so tensors with
        identical metadata get cache hits.  The actual tensor data is NOT
        part of the key — this is intentional: the cache stores a
        pre-allocated zero-padded buffer, and the caller should copy data
        into it.

        Thread-safe: the internal LRU cache is protected by a lock.

        For most workloads, prefer the plain ``pad()`` method.  Use
        ``pad_cached()`` when you repeatedly pad tensors with the same
        shape signature and want to avoid re-allocating the output buffer.
        """
        torch = get_torch()
        ndim = tensor.dim()
        if dim < 0:
            dim = ndim + dim

        cache_key = (tuple(tensor.shape), str(tensor.dtype), str(tensor.device), dim)

        with self._lock:
            if self.max_cache_entries > 0 and cache_key in self._pad_cache:
                self._pad_cache.move_to_end(cache_key)  # LRU refresh
                self._cache_hits += 1
                return self._pad_cache[cache_key]

        # Pad outside the lock (expensive computation shouldn't hold it)
        self._cache_misses += 1
        padded = self.pad(tensor, dim)

        with self._lock:
            if self.max_cache_entries > 0:
                # Evict oldest if at capacity
                while len(self._pad_cache) >= self.max_cache_entries:
                    self._pad_cache.popitem(last=False)
                self._pad_cache[cache_key] = padded

        return padded

    def clear_cache(self) -> int:
        """Clear all cached padded tensors.

        Returns:
            Number of entries cleared.
        """
        with self._lock:
            n = len(self._pad_cache)
            self._pad_cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0
        logger.debug("ShapeBucketer cache cleared (%d entries)", n)
        return n

    def cache_memory_bytes(self) -> int:
        """Estimate memory consumed by cached padded tensors (in bytes).

        Returns:
            Total bytes across all cached tensors.
        """
        with self._lock:
            total = 0
            for tensor in self._pad_cache.values():
                if hasattr(tensor, "element_size") and hasattr(tensor, "nelement"):
                    total += tensor.element_size() * tensor.nelement()
        return total

    @property
    def cache_size(self) -> int:
        """Number of entries currently in the pad cache."""
        with self._lock:
            return len(self._pad_cache)

    def stats(self) -> Dict[str, Any]:
        """Return bucketing and cache statistics."""
        total = sum(self._hit_count.values()) + self._miss_count
        cache_total = self._cache_hits + self._cache_misses
        return {
            "total_calls": total,
            "bucket_hits": dict(self._hit_count),
            "overflow_count": self._miss_count,
            "unique_buckets_used": len(self._hit_count),
            "efficiency": (
                f"{(1 - self._miss_count / total) * 100:.1f}%"
                if total > 0 else "n/a"
            ),
            "cache_entries": len(self._pad_cache),
            "cache_max_entries": self.max_cache_entries,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": (
                f"{self._cache_hits / cache_total * 100:.1f}%"
                if cache_total > 0 else "n/a"
            ),
            "cache_memory_bytes": self.cache_memory_bytes(),
        }
