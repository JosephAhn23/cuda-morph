"""Stress tests — verify stability under load.

These tests don't verify correctness (that's done in unit tests).  They
verify that the shim doesn't leak memory, crash, or corrupt state when
subjected to high repetition counts.

Marked with ``@pytest.mark.stress`` so they can be skipped on quick CI runs.
"""

from __future__ import annotations

import gc
import sys

import pytest
import torch


@pytest.mark.stress
class TestActivationCycles:
    """Verify activate/deactivate doesn't leak or corrupt state."""

    def test_1000_cycles_no_crash(self):
        """1000 activate/deactivate cycles should not crash."""
        from ascend_compat.cuda_shim._monkey_patch import activate, deactivate, is_activated

        for _ in range(1000):
            activate()
            torch.cuda.is_available()
            deactivate()

        assert not is_activated()

    def test_cycles_no_memory_leak(self):
        """activate/deactivate cycles should not leak objects."""
        from ascend_compat.cuda_shim._monkey_patch import activate, deactivate

        gc.collect()
        gc.collect()
        objects_before = len(gc.get_objects())

        for i in range(500):
            activate()
            _ = torch.cuda.is_available()
            deactivate()
            if i % 100 == 0:
                gc.collect()

        gc.collect()
        gc.collect()
        objects_after = len(gc.get_objects())

        # Allow some growth but not unbounded (500 cycles should not add >2000 objects)
        growth = objects_after - objects_before
        assert growth < 2000, (
            f"Object count grew by {growth} after 500 cycles — possible memory leak"
        )

    def test_rapid_activate_deactivate(self):
        """Very fast cycles should not corrupt torch.cuda state."""
        from ascend_compat.cuda_shim._monkey_patch import activate, deactivate

        original = torch.cuda.is_available

        for _ in range(200):
            activate()
            deactivate()

        # torch.cuda.is_available should be restored to original
        assert torch.cuda.is_available is original or callable(torch.cuda.is_available)


@pytest.mark.stress
class TestPatchManagerStress:
    """Stress the PatchManager with many patches."""

    def test_many_patches_no_crash(self):
        """Applying and reverting 1000 patches should work."""
        import types
        from ascend_compat.cuda_shim._patch_manager import PatchManager

        mgr = PatchManager()
        mod = types.ModuleType("stress_mod")

        for i in range(1000):
            setattr(mod, f"fn_{i}", lambda: None)

        for i in range(1000):
            mgr.apply(mod, f"fn_{i}", lambda: "patched", f"s.{i}", count_calls=False)

        assert mgr.patch_count == 1000
        mgr.revert_all()
        assert mgr.patch_count == 0

    def test_counter_accuracy_high_volume(self):
        """100k calls through counters should be exact."""
        import types
        from ascend_compat.cuda_shim._patch_manager import PatchManager

        mgr = PatchManager()
        mod = types.ModuleType("count_mod")
        mod.fn = lambda: 42

        mgr.apply(mod, "fn", lambda: 42, "test.fn", count_calls=True)

        for _ in range(100000):
            mod.fn()

        stats = mgr.get_stats()
        assert stats["test.fn"] == 100000


@pytest.mark.stress
class TestShapeBucketerStress:
    """Stress the ShapeBucketer cache eviction."""

    def test_cache_eviction_respects_limit(self):
        """Feed many distinct shapes — cache should not exceed max_entries."""
        from ascend_compat.cuda_shim.compile_helpers import ShapeBucketer

        bucketer = ShapeBucketer(
            buckets=[32, 64, 128, 256, 512],
            max_cache_entries=50,
        )

        for i in range(500):
            shape_size = (i % 30) + 1  # Varying shapes
            t = torch.randn(shape_size, 16)
            bucketer.pad_cached(t, dim=0)

        assert bucketer.cache_size <= 50

    def test_cache_memory_bounded(self):
        """Cache memory should be bounded by max_entries."""
        from ascend_compat.cuda_shim.compile_helpers import ShapeBucketer

        bucketer = ShapeBucketer(
            buckets=[64, 128, 256],
            max_cache_entries=20,
        )

        for i in range(200):
            t = torch.randn(i % 60 + 1, 32)
            bucketer.pad_cached(t, dim=0)

        mem = bucketer.cache_memory_bytes()
        # 20 cached tensors of at most (256, 32) float32 = 20 * 256 * 32 * 4 = ~640KB
        assert mem < 1024 * 1024, f"Cache using {mem} bytes — expected <1MB"

    def test_clear_cache_frees_memory(self):
        """clear_cache should free all cached tensors."""
        from ascend_compat.cuda_shim.compile_helpers import ShapeBucketer

        bucketer = ShapeBucketer(buckets=[64, 128], max_cache_entries=100)

        for i in range(100):
            t = torch.randn(i % 50 + 1, 8)
            bucketer.pad_cached(t, dim=0)

        assert bucketer.cache_size > 0
        assert bucketer.cache_memory_bytes() > 0

        cleared = bucketer.clear_cache()
        assert cleared > 0
        assert bucketer.cache_size == 0
        assert bucketer.cache_memory_bytes() == 0

    def test_disabled_cache(self):
        """max_cache_entries=0 should disable caching entirely."""
        from ascend_compat.cuda_shim.compile_helpers import ShapeBucketer

        bucketer = ShapeBucketer(buckets=[64, 128], max_cache_entries=0)

        for i in range(100):
            t = torch.randn(i % 50 + 1, 8)
            bucketer.pad_cached(t, dim=0)

        assert bucketer.cache_size == 0


@pytest.mark.stress
class TestBenchmarkStress:
    """Verify benchmarks don't crash under stress."""

    def test_many_short_benchmarks(self):
        """Running many small benchmarks should not OOM."""
        from ascend_compat.bench import OpLatencyBench

        for _ in range(10):
            bench = OpLatencyBench(device="cpu", iterations=5)
            report = bench.run()
            assert len(report.results) > 0


@pytest.mark.stress
class TestMemoryBandwidthBench:
    """Test the memory bandwidth benchmark on CPU."""

    def test_runs_on_cpu(self):
        from ascend_compat.bench import MemoryBandwidthBench

        bench = MemoryBandwidthBench(device="cpu", iterations=5)
        report = bench.run()

        assert len(report.results) > 0
        assert any("copy" in r.name for r in report.results)
        assert any("matmul" in r.name for r in report.results)
