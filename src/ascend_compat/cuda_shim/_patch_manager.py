"""Robust patch lifecycle manager with thread safety, reference counting, and telemetry.

This module solves three production-readiness problems at once:

1. **Thread safety** — ``activate()`` and ``deactivate()`` are protected by an
   RLock so concurrent threads cannot double-patch or leave partial state.

2. **Reference counting** — Multiple calls to ``activate()`` increment a counter;
   ``deactivate()`` only restores originals when the counter hits zero.  This
   prevents nested activate/deactivate pairs from clobbering each other.

3. **Atomic activation** — If any patch fails mid-activation, all previously
   applied patches in that batch are rolled back.  The system never enters a
   half-patched state.

4. **Telemetry** — Every patch call is counted.  ``get_patch_stats()`` returns
   a dict of ``{patch_name: hit_count}`` for production observability.

Usage (internal — called by ``_monkey_patch.py``)::

    from ascend_compat.cuda_shim._patch_manager import PatchManager

    mgr = PatchManager()
    mgr.apply(torch.cuda, "is_available", my_replacement, "cuda.is_available")
    mgr.revert_all()

    # Check how many times each patched function was called:
    stats = mgr.get_stats()
    # => {"cuda.is_available": 42, "torch.device": 137, ...}
"""

from __future__ import annotations

import threading
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

from ascend_compat._logging import get_logger

logger = get_logger(__name__)


class PatchRecord:
    """Record of a single applied patch."""

    __slots__ = ("module", "attr", "original", "patch_name")

    def __init__(self, module: Any, attr: str, original: Any, patch_name: str) -> None:
        self.module = module
        self.attr = attr
        self.original = original
        self.patch_name = patch_name


class PatchManager:
    """Thread-safe patch lifecycle manager with reference counting and telemetry.

    A PatchManager instance tracks all monkey-patches applied during activation.
    It provides:

    - ``apply(module, attr, replacement, name)`` — apply a single patch with
      automatic original-value backup and call counting.
    - ``revert_all()`` — restore all originals in reverse application order.
    - ``get_stats()`` — return per-patch call counters.

    Thread safety
    -------------
    All public methods acquire ``self._lock`` (an ``RLock`` — reentrant so the
    same thread can safely call ``apply`` from within a patched function).

    Atomic batches
    --------------
    Use ``begin_batch()`` / ``commit_batch()`` / ``rollback_batch()`` to group
    patches.  If any patch in a batch fails, call ``rollback_batch()`` to undo
    only the patches applied since ``begin_batch()``.

    Reference counting
    ------------------
    ``increment_ref()`` and ``decrement_ref()`` manage an activation counter.
    ``should_activate()`` returns True only when transitioning from 0→1.
    ``should_deactivate()`` returns True only when transitioning from 1→0.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._patches: List[PatchRecord] = []
        self._patch_keys: Dict[Tuple[int, str], int] = {}  # (id(module), attr) → index
        self._counters: Counter[str] = Counter()
        self._ref_count: int = 0
        self._batch_start: Optional[int] = None

    # ------------------------------------------------------------------
    # Reference counting
    # ------------------------------------------------------------------

    def increment_ref(self) -> bool:
        """Increment activation reference count.

        Returns True if this is the FIRST activation (count goes 0→1),
        meaning the caller should proceed with patching.  Returns False
        if already activated (count goes N→N+1).
        """
        with self._lock:
            was_zero = self._ref_count == 0
            self._ref_count += 1
            if was_zero:
                logger.debug("PatchManager: first activation (ref_count=1)")
            else:
                logger.debug("PatchManager: ref_count incremented to %d", self._ref_count)
            return was_zero

    def decrement_ref(self) -> bool:
        """Decrement activation reference count.

        Returns True if the count reached zero (caller should revert patches).
        Returns False if count is still positive.
        """
        with self._lock:
            if self._ref_count <= 0:
                logger.warning("PatchManager: decrement_ref called with ref_count=%d", self._ref_count)
                return False
            self._ref_count -= 1
            hit_zero = self._ref_count == 0
            if hit_zero:
                logger.debug("PatchManager: last deactivation (ref_count=0)")
            else:
                logger.debug("PatchManager: ref_count decremented to %d", self._ref_count)
            return hit_zero

    @property
    def ref_count(self) -> int:
        """Current activation reference count."""
        with self._lock:
            return self._ref_count

    @property
    def is_active(self) -> bool:
        """True if at least one activation is outstanding."""
        with self._lock:
            return self._ref_count > 0

    # ------------------------------------------------------------------
    # Patch application
    # ------------------------------------------------------------------

    def apply(
        self,
        module: Any,
        attr: str,
        replacement: Any,
        patch_name: str,
        *,
        count_calls: bool = True,
    ) -> None:
        """Apply a single patch, storing the original for later revert.

        Args:
            module: The module/object to patch (e.g. ``torch.cuda``).
            attr: The attribute name to replace (e.g. ``"is_available"``).
            replacement: The new callable/value to install.
            patch_name: Human-readable name for telemetry (e.g. ``"cuda.is_available"``).
            count_calls: If True and replacement is callable, wrap it to count calls.
        """
        with self._lock:
            key = (id(module), attr)

            # If already patched, just update the replacement (no new backup)
            if key in self._patch_keys:
                idx = self._patch_keys[key]
                record = self._patches[idx]
                if count_calls and callable(replacement):
                    replacement = self._wrap_with_counter(replacement, patch_name)
                setattr(module, attr, replacement)
                logger.debug("PatchManager: re-patched %s.%s (name=%s)", type(module).__name__, attr, patch_name)
                return

            # First time patching this (module, attr) — store original
            original = getattr(module, attr, None)

            if count_calls and callable(replacement):
                replacement = self._wrap_with_counter(replacement, patch_name)

            setattr(module, attr, replacement)

            record = PatchRecord(module, attr, original, patch_name)
            idx = len(self._patches)
            self._patches.append(record)
            self._patch_keys[key] = idx

            logger.debug(
                "PatchManager: applied %s.%s → %s (name=%s)",
                type(module).__name__, attr, getattr(replacement, "__name__", "?"), patch_name,
            )

    def _wrap_with_counter(self, fn: Callable[..., Any], patch_name: str) -> Callable[..., Any]:
        """Wrap a callable to increment its telemetry counter on each call."""
        counters = self._counters

        def counted(*args: Any, **kwargs: Any) -> Any:
            counters[patch_name] += 1
            return fn(*args, **kwargs)

        # Preserve function metadata
        counted.__name__ = getattr(fn, "__name__", patch_name)
        counted.__doc__ = getattr(fn, "__doc__", None)
        counted.__qualname__ = getattr(fn, "__qualname__", patch_name)
        counted._ascend_compat_patch = patch_name  # type: ignore[attr-defined]
        return counted

    # ------------------------------------------------------------------
    # Batch operations (atomic activation)
    # ------------------------------------------------------------------

    def begin_batch(self) -> None:
        """Mark the start of a batch.  Patches applied after this can be
        rolled back atomically with ``rollback_batch()``."""
        with self._lock:
            self._batch_start = len(self._patches)
            logger.debug("PatchManager: batch started at index %d", self._batch_start)

    def commit_batch(self) -> None:
        """Commit the current batch (no-op — just clears the marker)."""
        with self._lock:
            if self._batch_start is not None:
                n = len(self._patches) - self._batch_start
                logger.debug("PatchManager: batch committed (%d patches)", n)
            self._batch_start = None

    def rollback_batch(self) -> None:
        """Undo all patches applied since ``begin_batch()``.

        Restores originals in reverse order and removes the records.
        """
        with self._lock:
            if self._batch_start is None:
                logger.warning("PatchManager: rollback_batch called without begin_batch")
                return

            to_revert = self._patches[self._batch_start:]
            for record in reversed(to_revert):
                key = (id(record.module), record.attr)
                if record.original is not None:
                    setattr(record.module, record.attr, record.original)
                elif hasattr(record.module, record.attr):
                    delattr(record.module, record.attr)
                if key in self._patch_keys:
                    del self._patch_keys[key]
                logger.debug("PatchManager: rolled back %s (batch rollback)", record.patch_name)

            self._patches = self._patches[:self._batch_start]
            self._batch_start = None
            logger.info("PatchManager: batch rolled back")

    # ------------------------------------------------------------------
    # Revert
    # ------------------------------------------------------------------

    def revert_all(self) -> None:
        """Restore all originals in reverse application order."""
        with self._lock:
            for record in reversed(self._patches):
                key = (id(record.module), record.attr)
                if record.original is not None:
                    setattr(record.module, record.attr, record.original)
                elif hasattr(record.module, record.attr):
                    delattr(record.module, record.attr)
                if key in self._patch_keys:
                    del self._patch_keys[key]
                logger.debug("PatchManager: reverted %s", record.patch_name)

            n = len(self._patches)
            self._patches.clear()
            self._patch_keys.clear()
            self._batch_start = None
            logger.info("PatchManager: reverted %d patches", n)

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, int]:
        """Return per-patch call counters.

        Returns:
            Dict mapping patch name → number of times the patch was called.
            Only includes patches that have been called at least once.
        """
        with self._lock:
            return dict(self._counters)

    def get_all_stats(self) -> Dict[str, int]:
        """Return counters for ALL registered patches (including zero-count)."""
        with self._lock:
            result: Dict[str, int] = {}
            for record in self._patches:
                result[record.patch_name] = self._counters.get(record.patch_name, 0)
            return result

    def reset_stats(self) -> None:
        """Reset all counters to zero."""
        with self._lock:
            self._counters.clear()

    @property
    def patch_count(self) -> int:
        """Number of currently applied patches."""
        with self._lock:
            return len(self._patches)

    def get_patch_names(self) -> List[str]:
        """Return names of all currently applied patches."""
        with self._lock:
            return [r.patch_name for r in self._patches]
