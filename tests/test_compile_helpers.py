"""Tests for ascend_compat.cuda_shim.compile_helpers."""

from __future__ import annotations

import pytest


class TestGetCompileBackend:
    """Test compile backend detection."""

    def test_returns_string(self):
        from ascend_compat.cuda_shim.compile_helpers import get_compile_backend

        backend = get_compile_backend()
        assert isinstance(backend, str)
        assert len(backend) > 0

    def test_cpu_returns_inductor(self):
        """On a system without NPU, should return inductor or eager."""
        from ascend_compat.cuda_shim.compile_helpers import get_compile_backend

        backend = get_compile_backend()
        # Without NPU hardware, should be either inductor or eager
        assert backend in ("inductor", "eager", "torchair", "ascend")


class TestIsTorchairAvailable:
    """Test torchair detection."""

    def test_returns_bool(self):
        from ascend_compat.cuda_shim.compile_helpers import is_torchair_available

        result = is_torchair_available()
        assert isinstance(result, bool)
        # In test environment without Ascend, should be False
        # (but don't assert that — CI might have it)


class TestGetCompileInfo:
    """Test compile info diagnostic."""

    def test_returns_dict(self):
        from ascend_compat.cuda_shim.compile_helpers import get_compile_info

        info = get_compile_info()
        assert isinstance(info, dict)
        assert "recommended_backend" in info
        assert "torchair_available" in info
        assert "available_backends" in info

    def test_available_backends_is_list(self):
        from ascend_compat.cuda_shim.compile_helpers import get_compile_info

        info = get_compile_info()
        assert isinstance(info["available_backends"], list)


class TestEnableGraphMode:
    """Test graph mode management."""

    def test_enable_on_cpu_returns_false(self):
        """On CPU-only system, graph mode enable should return False."""
        from ascend_compat.cuda_shim.compile_helpers import enable_graph_mode

        result = enable_graph_mode()
        assert result is False

    def test_disable_on_cpu_returns_false(self):
        from ascend_compat.cuda_shim.compile_helpers import disable_graph_mode

        result = disable_graph_mode()
        assert result is False


class TestShapeBucketer:
    """Test shape bucketing for dynamic shapes."""

    def test_default_buckets(self):
        from ascend_compat.cuda_shim.compile_helpers import ShapeBucketer

        b = ShapeBucketer()
        assert len(b.buckets) > 0
        assert b.buckets == sorted(b.buckets)

    def test_bucket_size_rounds_up(self):
        from ascend_compat.cuda_shim.compile_helpers import ShapeBucketer

        b = ShapeBucketer(buckets=[128, 256, 512, 1024])

        assert b.bucket_size(100) == 128
        assert b.bucket_size(128) == 128
        assert b.bucket_size(129) == 256
        assert b.bucket_size(256) == 256
        assert b.bucket_size(500) == 512
        assert b.bucket_size(513) == 1024

    def test_bucket_size_exceeds_max(self):
        from ascend_compat.cuda_shim.compile_helpers import ShapeBucketer

        b = ShapeBucketer(buckets=[128, 256], max_size=512)
        assert b.bucket_size(300) == 512

    def test_predefined_strategies(self):
        from ascend_compat.cuda_shim.compile_helpers import ShapeBucketer

        assert len(ShapeBucketer.POWER_OF_2) > 0
        assert len(ShapeBucketer.LLM_INFERENCE) > 0
        assert len(ShapeBucketer.VISION) > 0
        assert ShapeBucketer.POWER_OF_2 == sorted(ShapeBucketer.POWER_OF_2)

    def test_pad_tensor(self):
        import torch
        from ascend_compat.cuda_shim.compile_helpers import ShapeBucketer

        b = ShapeBucketer(buckets=[128, 256, 512])

        # Create a 1D tensor with shape [100]
        t = torch.ones(100)
        padded = b.pad(t, dim=0)
        assert padded.shape[0] == 128
        # Original data preserved
        assert torch.all(padded[:100] == 1.0)
        # Padding is zeros
        assert torch.all(padded[100:] == 0.0)

    def test_pad_2d_tensor_dim0(self):
        import torch
        from ascend_compat.cuda_shim.compile_helpers import ShapeBucketer

        b = ShapeBucketer(buckets=[128, 256])

        t = torch.ones(100, 64)
        padded = b.pad(t, dim=0)
        assert padded.shape == (128, 64)
        assert torch.all(padded[:100, :] == 1.0)

    def test_pad_2d_tensor_dim1(self):
        import torch
        from ascend_compat.cuda_shim.compile_helpers import ShapeBucketer

        b = ShapeBucketer(buckets=[128, 256])

        t = torch.ones(32, 200)
        padded = b.pad(t, dim=1)
        assert padded.shape == (32, 256)
        assert torch.all(padded[:, :200] == 1.0)

    def test_pad_negative_dim(self):
        import torch
        from ascend_compat.cuda_shim.compile_helpers import ShapeBucketer

        b = ShapeBucketer(buckets=[128, 256])

        t = torch.ones(32, 100)
        padded = b.pad(t, dim=-1)  # last dim
        assert padded.shape == (32, 128)

    def test_pad_no_padding_needed(self):
        import torch
        from ascend_compat.cuda_shim.compile_helpers import ShapeBucketer

        b = ShapeBucketer(buckets=[128, 256])

        t = torch.ones(128)
        padded = b.pad(t, dim=0)
        # Should return the same tensor (no padding needed)
        assert padded.shape[0] == 128

    def test_stats(self):
        from ascend_compat.cuda_shim.compile_helpers import ShapeBucketer

        b = ShapeBucketer(buckets=[128, 256, 512])

        b.bucket_size(100)  # → 128
        b.bucket_size(100)  # → 128
        b.bucket_size(200)  # → 256
        b.bucket_size(600)  # overflow

        stats = b.stats()
        assert stats["total_calls"] == 4
        assert stats["bucket_hits"][128] == 2
        assert stats["bucket_hits"][256] == 1
        assert stats["overflow_count"] == 1
        assert stats["unique_buckets_used"] == 2

    def test_3d_tensor_pad(self):
        """Verify padding works on 3D tensors (batch, seq_len, hidden)."""
        import torch
        from ascend_compat.cuda_shim.compile_helpers import ShapeBucketer

        b = ShapeBucketer(buckets=[128, 256, 512])

        # batch=4, seq_len=137, hidden=768
        t = torch.randn(4, 137, 768)
        padded = b.pad(t, dim=1)
        assert padded.shape == (4, 256, 768)
        assert torch.allclose(padded[:, :137, :], t)
