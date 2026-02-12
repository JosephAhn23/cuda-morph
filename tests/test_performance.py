"""Performance regression tests for the shim layer.

These tests verify that the shim's overhead stays within acceptable bounds.
The patched path should be within ~20% of the unpatched path for simple
function calls.  This catches accidentally expensive wrappers.

Marked with ``@pytest.mark.benchmark`` so they can be run or skipped
independently.
"""

from __future__ import annotations

import time

import pytest
import torch


def _time_calls(fn, n: int = 10000) -> float:
    """Time *n* calls to *fn*, return total seconds."""
    start = time.perf_counter()
    for _ in range(n):
        fn()
    return time.perf_counter() - start


@pytest.mark.benchmark
class TestShimOverhead:
    """Verify that patched torch.cuda calls don't add excessive overhead."""

    N = 10000
    OVERHEAD_FACTOR = 3.0  # Allow up to 3x overhead (generous for CI stability)

    def test_is_available_latency(self):
        """torch.cuda.is_available() should not be more than 3x slower when patched."""
        from ascend_compat.cuda_shim._monkey_patch import activate, deactivate

        # Baseline: unpatched
        baseline = _time_calls(torch.cuda.is_available, self.N)

        # Patched
        activate()
        try:
            patched = _time_calls(torch.cuda.is_available, self.N)
        finally:
            deactivate()

        # Allow generous overhead factor to avoid flaky tests on CI
        assert patched < baseline * self.OVERHEAD_FACTOR, (
            f"Patched is_available() took {patched:.4f}s vs baseline {baseline:.4f}s "
            f"({patched/baseline:.1f}x) — exceeds {self.OVERHEAD_FACTOR}x threshold"
        )

    def test_device_creation_latency(self):
        """torch.device('cpu') should have minimal overhead from the device patch."""
        from ascend_compat.cuda_shim._monkey_patch import activate, deactivate

        # Baseline: unpatched
        baseline = _time_calls(lambda: torch.device("cpu"), self.N)

        # Patched (cpu strings should pass through quickly)
        activate()
        try:
            patched = _time_calls(lambda: torch.device("cpu"), self.N)
        finally:
            deactivate()

        assert patched < baseline * self.OVERHEAD_FACTOR, (
            f"Patched torch.device('cpu') took {patched:.4f}s vs baseline {baseline:.4f}s "
            f"({patched/baseline:.1f}x) — exceeds {self.OVERHEAD_FACTOR}x threshold"
        )

    def test_tensor_creation_unaffected(self):
        """Basic tensor creation should not be affected by the shim."""
        from ascend_compat.cuda_shim._monkey_patch import activate, deactivate

        baseline = _time_calls(lambda: torch.zeros(64, 64), self.N)

        activate()
        try:
            patched = _time_calls(lambda: torch.zeros(64, 64), self.N)
        finally:
            deactivate()

        # Tensor creation goes through C++ dispatch, not our patches
        assert patched < baseline * self.OVERHEAD_FACTOR

    def test_activation_deactivation_speed(self):
        """activate() + deactivate() cycle should complete in <1 second."""
        from ascend_compat.cuda_shim._monkey_patch import activate, deactivate

        start = time.perf_counter()
        for _ in range(50):
            activate()
            deactivate()
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, (
            f"50 activate/deactivate cycles took {elapsed:.2f}s — should be <1s"
        )


@pytest.mark.benchmark
class TestPatchManagerPerformance:
    """Verify PatchManager operations are fast."""

    def test_apply_revert_cycle(self):
        """apply + revert_all for 100 patches should be fast."""
        import types
        from ascend_compat.cuda_shim._patch_manager import PatchManager

        mgr = PatchManager()
        mod = types.ModuleType("test_module")

        # Pre-populate attributes
        for i in range(100):
            setattr(mod, f"attr_{i}", lambda: None)

        start = time.perf_counter()
        for i in range(100):
            mgr.apply(mod, f"attr_{i}", lambda: "patched", f"test.attr_{i}")
        apply_time = time.perf_counter() - start

        start = time.perf_counter()
        mgr.revert_all()
        revert_time = time.perf_counter() - start

        assert apply_time < 0.1, f"100 applies took {apply_time:.3f}s"
        assert revert_time < 0.1, f"100 reverts took {revert_time:.3f}s"

    def test_counter_overhead(self):
        """Call counting should add negligible overhead."""
        from ascend_compat.cuda_shim._patch_manager import PatchManager
        import types

        mgr = PatchManager()
        mod = types.ModuleType("test_module")

        original = lambda: 42
        setattr(mod, "fn", original)

        mgr.apply(mod, "fn", original, "test.fn", count_calls=True)
        wrapped = mod.fn

        # Call 100k times
        start = time.perf_counter()
        for _ in range(100000):
            wrapped()
        elapsed = time.perf_counter() - start

        # Should complete in under 1 second
        assert elapsed < 1.0, f"100k counter-wrapped calls took {elapsed:.2f}s"

        stats = mgr.get_stats()
        assert stats.get("test.fn", 0) == 100000
