"""Shared test fixtures for ascend-compat.

These fixtures create isolated test environments that simulate different
hardware configurations (CPU-only, CUDA, NPU) without requiring actual
hardware.  This is essential for CI/CD testing.

Design decisions:
- We mock at the ascend_compat._backend level (not torch level) so we
  control what the shim *thinks* is available
- Each fixture clears cached backend detection to ensure clean state
- The cuda_shim is deactivated before each test to prevent cross-contamination
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ascend_compat._backend import Backend


def _clear_backend_caches() -> None:
    """Clear all cached backend detection state."""
    from ascend_compat import _backend

    # Clear lru_cache on detection functions
    _backend.detect_backends.cache_clear()
    _backend.preferred_backend.cache_clear()

    # Reset lazy import state
    _backend._torch = None
    _backend._torch_npu = None


def _reset_patch_manager() -> None:
    """Reset the PatchManager to a clean state.

    This replaces the old approach of clearing ``_originals`` / ``_activated``.
    The PatchManager's internal state is cleared by reverting all patches
    and resetting the reference count.
    """
    from ascend_compat.cuda_shim._monkey_patch import _manager
    from ascend_compat.cuda_shim._import_hook import uninstall_import_hook

    # Revert all patches (restores originals)
    _manager.revert_all()

    # Force reference count to 0
    _manager._ref_count = 0

    # Reset telemetry
    _manager.reset_stats()

    # Uninstall import hook
    uninstall_import_hook()


@pytest.fixture(autouse=True)
def _reset_shim_state():
    """Ensure clean state before every test."""
    _reset_patch_manager()
    _clear_backend_caches()
    yield
    _reset_patch_manager()
    _clear_backend_caches()


@pytest.fixture
def cpu_only_backend():
    """Simulate a CPU-only environment (no GPU, no NPU)."""
    with patch("ascend_compat._backend.detect_backends",
               return_value=(Backend.CPU,)) as mock_detect:
        mock_detect.cache_clear = lambda: None
        with patch("ascend_compat._backend.preferred_backend",
                    return_value=Backend.CPU) as mock_pref:
            mock_pref.cache_clear = lambda: None
            yield


@pytest.fixture
def cuda_backend():
    """Simulate an NVIDIA CUDA environment."""
    with patch("ascend_compat._backend.detect_backends",
               return_value=(Backend.CUDA, Backend.CPU)) as mock_detect:
        mock_detect.cache_clear = lambda: None
        with patch("ascend_compat._backend.preferred_backend",
                    return_value=Backend.CUDA) as mock_pref:
            mock_pref.cache_clear = lambda: None
            yield


@pytest.fixture
def npu_backend():
    """Simulate an Ascend NPU environment."""
    with patch("ascend_compat._backend.detect_backends",
               return_value=(Backend.NPU, Backend.CPU)) as mock_detect:
        mock_detect.cache_clear = lambda: None
        with patch("ascend_compat._backend.preferred_backend",
                    return_value=Backend.NPU) as mock_pref:
            mock_pref.cache_clear = lambda: None
            with patch("ascend_compat._backend.has_npu", return_value=True):
                yield
