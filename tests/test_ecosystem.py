"""Tests for the ecosystem layer (flash_attn, transformers_patch, deepspeed_patch)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from ascend_compat.ecosystem.flash_attn import (
    _get_npu_fusion_attention,
    flash_attn_func,
    flash_attn_with_kvcache,
)


class TestFlashAttnShim:
    """Tests for the flash_attn compatibility shim."""

    def test_raises_without_torch_npu(self) -> None:
        """flash_attn_func should raise RuntimeError when torch_npu is absent."""
        import torch
        q = torch.randn(1, 8, 4, 64)
        k = torch.randn(1, 8, 4, 64)
        v = torch.randn(1, 8, 4, 64)

        with pytest.raises(RuntimeError, match="npu_fusion_attention not available"):
            flash_attn_func(q, k, v)

    def test_calls_npu_fusion_attention(self) -> None:
        """Verify argument translation when calling the NPU backend."""
        import torch

        q = torch.randn(2, 16, 4, 64)
        k = torch.randn(2, 16, 4, 64)
        v = torch.randn(2, 16, 4, 64)

        mock_output = torch.randn(2, 16, 4, 64)
        mock_fn = MagicMock(return_value=(
            mock_output,
            torch.zeros(2, 4),   # softmax_max
            torch.ones(2, 4),    # softmax_sum
            torch.zeros(2, 4),   # softmax_out
        ))

        with patch(
            "ascend_compat.ecosystem.flash_attn._get_npu_fusion_attention",
            return_value=mock_fn,
        ):
            result = flash_attn_func(q, k, v, dropout_p=0.1, causal=True)

        assert result.shape == mock_output.shape
        mock_fn.assert_called_once()

        # Verify key argument translations
        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["head_num"] == 4
        assert call_kwargs["input_layout"] == "BSND"
        assert call_kwargs["keep_prob"] == pytest.approx(0.9)  # 1 - dropout_p
        assert call_kwargs["next_tockens"] == 0  # causal=True

    def test_causal_false_sets_large_next_tokens(self) -> None:
        """causal=False should set next_tockens to a very large number."""
        import torch

        q = torch.randn(1, 8, 2, 32)
        k = torch.randn(1, 8, 2, 32)
        v = torch.randn(1, 8, 2, 32)

        mock_fn = MagicMock(return_value=(
            torch.randn(1, 8, 2, 32),
            torch.zeros(1), torch.ones(1), torch.zeros(1),
        ))

        with patch(
            "ascend_compat.ecosystem.flash_attn._get_npu_fusion_attention",
            return_value=mock_fn,
        ):
            flash_attn_func(q, k, v, causal=False)

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["next_tockens"] == 2147483647

    def test_return_attn_probs(self) -> None:
        """return_attn_probs=True should return a tuple of 3."""
        import torch

        q = torch.randn(1, 4, 2, 32)
        k = torch.randn(1, 4, 2, 32)
        v = torch.randn(1, 4, 2, 32)

        mock_fn = MagicMock(return_value=(
            torch.randn(1, 4, 2, 32),
            torch.ones(1, 2),    # softmax_max
            torch.ones(1, 2),    # softmax_sum
            torch.zeros(1, 2),
        ))

        with patch(
            "ascend_compat.ecosystem.flash_attn._get_npu_fusion_attention",
            return_value=mock_fn,
        ):
            result = flash_attn_func(q, k, v, return_attn_probs=True)

        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_softmax_scale_default(self) -> None:
        """Default softmax_scale should be 1/sqrt(headdim)."""
        import math
        import torch

        headdim = 64
        q = torch.randn(1, 4, 2, headdim)
        k = torch.randn(1, 4, 2, headdim)
        v = torch.randn(1, 4, 2, headdim)

        mock_fn = MagicMock(return_value=(
            torch.randn(1, 4, 2, headdim),
            torch.zeros(1), torch.ones(1), torch.zeros(1),
        ))

        with patch(
            "ascend_compat.ecosystem.flash_attn._get_npu_fusion_attention",
            return_value=mock_fn,
        ):
            flash_attn_func(q, k, v)

        expected_scale = 1.0 / math.sqrt(headdim)
        actual_scale = mock_fn.call_args[1]["scale"]
        assert actual_scale == pytest.approx(expected_scale)


class TestFlashAttnPackageRegistration:
    """Test that flash_attn shim can be registered as a package."""

    def test_register_as_package(self) -> None:
        """Registering the shim should make `from flash_attn import flash_attn_func` work."""
        # Remove any existing flash_attn from sys.modules
        for key in list(sys.modules):
            if key.startswith("flash_attn"):
                del sys.modules[key]

        from ascend_compat.ecosystem import flash_attn as fa_shim
        sys.modules["flash_attn"] = fa_shim  # type: ignore[assignment]

        try:
            from flash_attn import flash_attn_func as imported_func  # type: ignore[import-untyped]
            assert imported_func is fa_shim.flash_attn_func
        finally:
            # Clean up
            for key in list(sys.modules):
                if key == "flash_attn" or key.startswith("flash_attn."):
                    del sys.modules[key]


class TestTransformersPatch:
    """Tests for the HuggingFace Transformers compatibility patch."""

    def test_apply_without_transformers(self) -> None:
        """apply() should not crash if transformers is not installed."""
        from ascend_compat.ecosystem.transformers_patch import apply

        # Force has_npu to return True
        with patch("ascend_compat.ecosystem.transformers_patch.has_npu", return_value=True):
            apply()  # Should not raise (gracefully skips)

    def test_apply_without_npu(self) -> None:
        """apply() should be a no-op without NPU."""
        from ascend_compat.ecosystem import transformers_patch
        # Reset state
        transformers_patch._applied = False

        with patch("ascend_compat.ecosystem.transformers_patch.has_npu", return_value=False):
            transformers_patch.apply()
            # Still not applied because no NPU
            assert not transformers_patch._applied


class TestDeepSpeedPatch:
    """Tests for the DeepSpeed compatibility patch."""

    def test_apply_without_deepspeed(self) -> None:
        """apply() should not crash if DeepSpeed is not installed."""
        from ascend_compat.ecosystem.deepspeed_patch import apply
        from ascend_compat.ecosystem import deepspeed_patch
        deepspeed_patch._applied = False

        with patch("ascend_compat.ecosystem.deepspeed_patch.has_npu", return_value=True):
            apply()  # Should not raise

    def test_visible_devices_mapping(self) -> None:
        """CUDA_VISIBLE_DEVICES should map to ASCEND_RT_VISIBLE_DEVICES."""
        import os
        from ascend_compat.ecosystem.deepspeed_patch import _patch_visible_devices_env

        old_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        old_ascend = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")

        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
            os.environ.pop("ASCEND_RT_VISIBLE_DEVICES", None)

            _patch_visible_devices_env()

            assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == "0,1,2"
        finally:
            if old_cuda is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda
            else:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            if old_ascend is not None:
                os.environ["ASCEND_RT_VISIBLE_DEVICES"] = old_ascend
            else:
                os.environ.pop("ASCEND_RT_VISIBLE_DEVICES", None)
