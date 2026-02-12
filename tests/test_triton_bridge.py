"""Tests for the Triton-Ascend integration bridge."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ascend_compat.ecosystem.triton_bridge import (
    configure_triton_backend,
    get_triton_info,
    is_triton_ascend_available,
    is_triton_available,
    triton_kernel_available,
)


class TestTritonAvailability:
    """Test Triton availability detection."""

    def test_triton_not_installed(self) -> None:
        """Should return False when triton is not installed."""
        with patch.dict("sys.modules", {"triton": None}):
            assert not is_triton_available()

    def test_triton_installed(self) -> None:
        """Should return True when triton can be imported."""
        mock_triton = MagicMock()
        mock_triton.__version__ = "3.0.0"
        with patch.dict("sys.modules", {"triton": mock_triton}):
            assert is_triton_available()

    def test_ascend_backend_not_available(self) -> None:
        """Should return False when no Ascend backend is registered."""
        mock_triton = MagicMock()
        mock_triton.backends = MagicMock(spec=[])  # Empty spec = no attributes
        del mock_triton.backends.ascend  # Explicitly remove
        del mock_triton.backends.npu
        with patch.dict("sys.modules", {"triton": mock_triton, "triton_ascend": None}):
            assert not is_triton_ascend_available()


class TestTritonInfo:
    """Test Triton info collection."""

    def test_info_without_triton(self) -> None:
        with patch.dict("sys.modules", {"triton": None}):
            info = get_triton_info()
        assert info["triton_installed"] is False
        assert info["triton_version"] is None

    def test_info_with_triton(self) -> None:
        mock_triton = MagicMock()
        mock_triton.__version__ = "2.5.0"
        mock_triton.backends = MagicMock()
        # Provide some backend names
        mock_triton.backends.__dir__ = lambda self: ["cuda", "rocm"]
        with patch.dict("sys.modules", {"triton": mock_triton}):
            with patch("ascend_compat.ecosystem.triton_bridge.is_triton_ascend_available",
                       return_value=False):
                info = get_triton_info()
        assert info["triton_installed"] is True
        assert info["triton_version"] == "2.5.0"


class TestConfigureBackend:
    """Test Triton backend configuration."""

    def test_warns_when_not_available(self) -> None:
        with patch("ascend_compat.ecosystem.triton_bridge.is_triton_ascend_available",
                   return_value=False):
            result = configure_triton_backend()
        assert result is False

    def test_configures_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TRITON_BACKEND", raising=False)
        with patch("ascend_compat.ecosystem.triton_bridge.is_triton_ascend_available",
                   return_value=True):
            result = configure_triton_backend()
        assert result is True
        import os
        assert os.environ.get("TRITON_BACKEND") == "ascend"


class TestKernelCompatibility:
    """Test kernel compatibility checking."""

    def test_warp_shuffle_incompatible(self) -> None:
        assert not triton_kernel_available("custom_warp_shuffle_kernel")

    def test_tensor_core_incompatible(self) -> None:
        assert not triton_kernel_available("tensor_core_gemm")

    def test_cooperative_group_incompatible(self) -> None:
        assert not triton_kernel_available("cooperative_group_reduce")

    def test_generic_matmul_compatible(self) -> None:
        assert triton_kernel_available("matmul_kernel")

    def test_flash_attention_compatible(self) -> None:
        assert triton_kernel_available("flash_attention_fwd")

    def test_elementwise_compatible(self) -> None:
        assert triton_kernel_available("fused_add_relu")
