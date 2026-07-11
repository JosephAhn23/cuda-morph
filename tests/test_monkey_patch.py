"""Tests for the monkey-patch layer (cuda_shim._monkey_patch)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from ascend_compat._backend import Backend
from ascend_compat.cuda_shim._monkey_patch import (
    _make_proxy,
    _make_unsupported_stub,
    activate,
    deactivate,
    is_activated,
)


class TestActivation:
    """Tests for shim activation/deactivation."""

    def test_activate_deactivate_cycle(self, cpu_only_backend: None) -> None:
        """activate() + deactivate() should be a clean cycle."""
        assert not is_activated()
        activate()
        assert is_activated()
        deactivate()
        assert not is_activated()

    def test_activate_is_idempotent(self, cpu_only_backend: None) -> None:
        activate()
        activate()  # Should not raise or double-patch (ref_count=2)
        assert is_activated()
        deactivate()  # ref_count=1
        deactivate()  # ref_count=0

    def test_no_patch_env_var(self, cpu_only_backend: None) -> None:
        """ASCEND_COMPAT_NO_PATCH=1 should skip activation."""
        import os
        old = os.environ.get("ASCEND_COMPAT_NO_PATCH")
        try:
            os.environ["ASCEND_COMPAT_NO_PATCH"] = "1"
            activate()
            assert not is_activated()
        finally:
            if old is None:
                os.environ.pop("ASCEND_COMPAT_NO_PATCH", None)
            else:
                os.environ["ASCEND_COMPAT_NO_PATCH"] = old


class TestCPUFallback:
    """Tests for CPU-only fallback patching."""

    def test_cpu_is_available_returns_false(self, cpu_only_backend: None) -> None:
        """On CPU-only, torch.cuda.is_available() should return False."""
        activate()
        assert torch.cuda.is_available() is False
        deactivate()

    def test_cpu_device_count_returns_zero(self, cpu_only_backend: None) -> None:
        activate()
        assert torch.cuda.device_count() == 0
        deactivate()

    def test_cpu_memory_returns_zero(self, cpu_only_backend: None) -> None:
        activate()
        assert torch.cuda.memory_allocated() == 0
        assert torch.cuda.max_memory_allocated() == 0
        deactivate()

    def test_cpu_empty_cache_noop(self, cpu_only_backend: None) -> None:
        """empty_cache should be a safe no-op on CPU."""
        activate()
        torch.cuda.empty_cache()  # Should not raise
        deactivate()

    def test_cpu_manual_seed_noop(self, cpu_only_backend: None) -> None:
        activate()
        torch.cuda.manual_seed(42)  # Should not raise
        torch.cuda.manual_seed_all(42)
        deactivate()


class TestProxyAndStub:
    """Tests for the proxy and unsupported stub factories."""

    def test_make_proxy_delegates(self) -> None:
        """_make_proxy should delegate to the target function."""
        target = MagicMock(return_value=42)
        proxy = _make_proxy(target, "test_attr", "npu_test_attr")
        result = proxy(1, 2, key="val")
        assert result == 42
        target.assert_called_once_with(1, 2, key="val")

    def test_make_proxy_has_name(self) -> None:
        target = MagicMock()
        proxy = _make_proxy(target, "my_func", "npu_my_func")
        assert proxy.__name__ == "my_func"

    def test_unsupported_stub_raises(self) -> None:
        """Unsupported stubs should raise NotImplementedError."""
        stub = _make_unsupported_stub("fake_op", "Use alternative X")
        import pytest
        with pytest.raises(NotImplementedError, match="fake_op"):
            stub()

    def test_unsupported_stub_includes_note(self) -> None:
        stub = _make_unsupported_stub("fake_op", "Use alternative X")
        import pytest
        with pytest.raises(NotImplementedError, match="Use alternative X"):
            stub()
