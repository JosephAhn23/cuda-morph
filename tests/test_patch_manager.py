"""Tests for ascend_compat.cuda_shim._patch_manager — PatchManager."""

from __future__ import annotations

import threading
import types

import pytest

from ascend_compat.cuda_shim._patch_manager import PatchManager


@pytest.fixture
def mgr():
    """Fresh PatchManager for each test."""
    return PatchManager()


@pytest.fixture
def mod():
    """Disposable module with some attributes."""
    m = types.ModuleType("test_mod")
    m.fn_a = lambda: "original_a"
    m.fn_b = lambda: "original_b"
    m.fn_c = lambda: "original_c"
    return m


class TestBasicPatchLifecycle:
    """Test basic apply/revert behavior."""

    def test_apply_replaces_attribute(self, mgr, mod):
        mgr.apply(mod, "fn_a", lambda: "patched", "test.fn_a", count_calls=False)
        assert mod.fn_a() == "patched"

    def test_revert_restores_original(self, mgr, mod):
        original = mod.fn_a
        mgr.apply(mod, "fn_a", lambda: "patched", "test.fn_a", count_calls=False)
        assert mod.fn_a() == "patched"

        mgr.revert_all()
        assert mod.fn_a is original

    def test_multiple_patches_reverted_in_order(self, mgr, mod):
        orig_a = mod.fn_a
        orig_b = mod.fn_b

        mgr.apply(mod, "fn_a", lambda: "patched_a", "a", count_calls=False)
        mgr.apply(mod, "fn_b", lambda: "patched_b", "b", count_calls=False)

        mgr.revert_all()
        assert mod.fn_a is orig_a
        assert mod.fn_b is orig_b

    def test_revert_empty_is_safe(self, mgr):
        mgr.revert_all()  # Should not raise

    def test_re_apply_same_attr_does_not_duplicate(self, mgr, mod):
        mgr.apply(mod, "fn_a", lambda: "v1", "a", count_calls=False)
        mgr.apply(mod, "fn_a", lambda: "v2", "a", count_calls=False)
        assert mod.fn_a() == "v2"
        assert mgr.patch_count == 1  # Not 2

    def test_patch_count(self, mgr, mod):
        assert mgr.patch_count == 0
        mgr.apply(mod, "fn_a", lambda: 1, "a", count_calls=False)
        assert mgr.patch_count == 1
        mgr.apply(mod, "fn_b", lambda: 2, "b", count_calls=False)
        assert mgr.patch_count == 2


class TestReferenceCount:
    """Test activation reference counting."""

    def test_first_increment_returns_true(self, mgr):
        assert mgr.increment_ref() is True

    def test_second_increment_returns_false(self, mgr):
        mgr.increment_ref()
        assert mgr.increment_ref() is False

    def test_decrement_to_zero_returns_true(self, mgr):
        mgr.increment_ref()
        assert mgr.decrement_ref() is True

    def test_decrement_not_to_zero_returns_false(self, mgr):
        mgr.increment_ref()
        mgr.increment_ref()
        assert mgr.decrement_ref() is False  # Still at 1
        assert mgr.decrement_ref() is True   # Now at 0

    def test_decrement_below_zero_returns_false(self, mgr):
        assert mgr.decrement_ref() is False

    def test_ref_count_property(self, mgr):
        assert mgr.ref_count == 0
        mgr.increment_ref()
        assert mgr.ref_count == 1
        mgr.increment_ref()
        assert mgr.ref_count == 2

    def test_is_active_property(self, mgr):
        assert mgr.is_active is False
        mgr.increment_ref()
        assert mgr.is_active is True
        mgr.decrement_ref()
        assert mgr.is_active is False


class TestAtomicBatches:
    """Test atomic batch operations."""

    def test_commit_preserves_patches(self, mgr, mod):
        mgr.begin_batch()
        mgr.apply(mod, "fn_a", lambda: "patched", "a", count_calls=False)
        mgr.commit_batch()

        assert mod.fn_a() == "patched"
        assert mgr.patch_count == 1

    def test_rollback_reverts_batch(self, mgr, mod):
        orig = mod.fn_a
        mgr.apply(mod, "fn_c", lambda: "pre", "c", count_calls=False)  # Before batch

        mgr.begin_batch()
        mgr.apply(mod, "fn_a", lambda: "in_batch", "a", count_calls=False)
        mgr.apply(mod, "fn_b", lambda: "in_batch", "b", count_calls=False)
        mgr.rollback_batch()

        # Batch patches reverted
        assert mod.fn_a is orig
        # Pre-batch patch preserved
        assert mod.fn_c() == "pre"
        assert mgr.patch_count == 1  # Only the pre-batch patch

    def test_rollback_without_begin_is_safe(self, mgr):
        mgr.rollback_batch()  # Should not raise

    def test_nested_batches_not_supported(self, mgr, mod):
        """Second begin_batch overwrites the first marker — this is by design."""
        mgr.begin_batch()
        mgr.apply(mod, "fn_a", lambda: "batch1", "a", count_calls=False)
        mgr.begin_batch()  # Overwrites marker
        mgr.apply(mod, "fn_b", lambda: "batch2", "b", count_calls=False)
        mgr.rollback_batch()

        # Only fn_b was in the "inner batch" — fn_a remains
        assert mod.fn_a() == "batch1"


class TestTelemetry:
    """Test call counting and stats."""

    def test_counted_calls(self, mgr, mod):
        mgr.apply(mod, "fn_a", lambda: "ok", "test.fn_a", count_calls=True)
        mod.fn_a()
        mod.fn_a()
        mod.fn_a()

        stats = mgr.get_stats()
        assert stats["test.fn_a"] == 3

    def test_uncounted_calls(self, mgr, mod):
        mgr.apply(mod, "fn_a", lambda: "ok", "test.fn_a", count_calls=False)
        mod.fn_a()
        mod.fn_a()

        stats = mgr.get_stats()
        assert "test.fn_a" not in stats

    def test_get_all_stats_includes_zero(self, mgr, mod):
        mgr.apply(mod, "fn_a", lambda: "ok", "a", count_calls=True)
        mgr.apply(mod, "fn_b", lambda: "ok", "b", count_calls=True)
        mod.fn_a()

        all_stats = mgr.get_all_stats()
        assert all_stats["a"] == 1
        assert all_stats["b"] == 0

    def test_reset_stats(self, mgr, mod):
        mgr.apply(mod, "fn_a", lambda: "ok", "a", count_calls=True)
        mod.fn_a()
        assert mgr.get_stats()["a"] == 1

        mgr.reset_stats()
        assert mgr.get_stats() == {}

    def test_get_patch_names(self, mgr, mod):
        mgr.apply(mod, "fn_a", lambda: 1, "alpha", count_calls=False)
        mgr.apply(mod, "fn_b", lambda: 2, "beta", count_calls=False)
        assert mgr.get_patch_names() == ["alpha", "beta"]


class TestThreadSafety:
    """Verify thread-safe behavior."""

    def test_concurrent_apply(self, mod):
        """Multiple threads applying patches should not corrupt state."""
        mgr = PatchManager()
        errors = []

        def worker(i: int):
            try:
                attr = f"fn_{i}"
                setattr(mod, attr, lambda: f"original_{i}")
                mgr.apply(mod, attr, lambda: f"patched_{i}", f"t.{i}", count_calls=False)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert mgr.patch_count == 20

    def test_concurrent_increment_decrement(self):
        """Reference count should be consistent under contention."""
        mgr = PatchManager()

        def increment_then_decrement():
            mgr.increment_ref()
            mgr.decrement_ref()

        threads = [threading.Thread(target=increment_then_decrement) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # After all threads complete, ref_count should be 0
        assert mgr.ref_count == 0

    def test_concurrent_stats_access(self, mod):
        """Reading stats while patches are being called should not crash."""
        mgr = PatchManager()
        mgr.apply(mod, "fn_a", lambda: 42, "a", count_calls=True)
        errors = []

        def call_fn():
            try:
                for _ in range(1000):
                    mod.fn_a()
            except Exception as e:
                errors.append(e)

        def read_stats():
            try:
                for _ in range(100):
                    mgr.get_stats()
            except Exception as e:
                errors.append(e)

        callers = [threading.Thread(target=call_fn) for _ in range(5)]
        readers = [threading.Thread(target=read_stats) for _ in range(3)]

        for t in callers + readers:
            t.start()
        for t in callers + readers:
            t.join()

        assert not errors
        assert mgr.get_stats()["a"] == 5000


class TestActivateDeactivateIntegration:
    """Test the actual activate/deactivate using PatchManager."""

    def test_activate_deactivate_preserves_originals(self):
        """After activate → deactivate, torch.cuda.is_available should work."""
        import torch
        from ascend_compat.cuda_shim._monkey_patch import activate, deactivate

        original_fn = torch.cuda.is_available
        original_result = original_fn()

        activate()
        # Should be patched now
        assert torch.cuda.is_available() is False

        deactivate()
        # Should be restored
        assert torch.cuda.is_available() == original_result

    def test_double_activate_single_deactivate(self):
        """Double activate + single deactivate should leave patches active."""
        from ascend_compat.cuda_shim._monkey_patch import activate, deactivate, is_activated

        activate()
        activate()  # ref_count = 2
        assert is_activated()

        deactivate()  # ref_count = 1
        assert is_activated()

        deactivate()  # ref_count = 0
        assert not is_activated()

    def test_stats_after_activation(self):
        """Patch stats should be available after activation."""
        import torch
        from ascend_compat.cuda_shim._monkey_patch import (
            activate, deactivate, get_patch_stats, reset_patch_stats,
        )

        activate()
        reset_patch_stats()

        # Call a patched function
        torch.cuda.is_available()
        torch.cuda.is_available()

        stats = get_patch_stats()
        # On CPU backend the patch name is "cpu.is_available",
        # on NPU it would be "cuda.is_available"
        is_avail_count = stats.get("cuda.is_available", 0) + stats.get("cpu.is_available", 0)
        assert is_avail_count >= 2

        deactivate()
