"""Tests for vllm-ascend compatibility patches."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from ascend_compat.ecosystem.vllm_patch import (
    SUPPORTED_QUANT_METHODS,
    UNSUPPORTED_QUANT_METHODS,
    _patch_visible_devices,
    _validate_cann_env,
    apply,
    check_vllm_readiness,
)


class TestVisibleDevices:
    """Test CUDA_VISIBLE_DEVICES → ASCEND_RT_VISIBLE_DEVICES mapping."""

    def test_maps_cuda_to_ascend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
        monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
        _patch_visible_devices()
        assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == "0,1,2"

    def test_does_not_overwrite_existing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
        monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "3,4")
        _patch_visible_devices()
        assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == "3,4"

    def test_noop_without_cuda_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
        _patch_visible_devices()
        assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") is None


class TestCANNEnvValidation:
    """Test CANN environment validation."""

    def test_reports_missing_ascend_home(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
        # Patch os.path.isdir to return False for all candidates
        with patch("os.path.isdir", return_value=False):
            issues = _validate_cann_env()
        assert "ASCEND_HOME_PATH" in issues

    def test_reports_missing_cmake(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ASCEND_HOME_PATH", "/nonexistent")
        with patch("shutil.which", return_value=None):
            with patch("os.path.isfile", return_value=False):
                issues = _validate_cann_env()
        assert "cmake" in issues


class TestApply:
    """Test the idempotent apply() function."""

    def test_skips_without_npu(self) -> None:
        import ascend_compat.ecosystem.vllm_patch as mod
        mod._applied = False
        with patch("ascend_compat.ecosystem.vllm_patch.has_npu", return_value=False):
            apply()
        assert mod._applied is False

    def test_applies_with_npu(self) -> None:
        import ascend_compat.ecosystem.vllm_patch as mod
        mod._applied = False
        with patch("ascend_compat.ecosystem.vllm_patch.has_npu", return_value=True):
            with patch("ascend_compat.ecosystem.vllm_patch._patch_visible_devices"):
                with patch("ascend_compat.ecosystem.vllm_patch._validate_cann_env", return_value={}):
                    with patch("ascend_compat.ecosystem.vllm_patch._patch_vllm_attention_backend"):
                        with patch("ascend_compat.ecosystem.vllm_patch._patch_vllm_quant_detection"):
                            apply()
        assert mod._applied is True
        mod._applied = False  # Reset for other tests


class TestQuantMethods:
    """Test quantization method classification."""

    def test_fp8_is_unsupported(self) -> None:
        assert "fp8" in UNSUPPORTED_QUANT_METHODS

    def test_awq_is_unsupported(self) -> None:
        assert "awq" in UNSUPPORTED_QUANT_METHODS

    def test_w8a8_is_supported(self) -> None:
        assert "w8a8" in SUPPORTED_QUANT_METHODS

    def test_none_is_supported(self) -> None:
        assert None in SUPPORTED_QUANT_METHODS


class TestVLLMReadiness:
    """Test the vLLM readiness checker."""

    def test_not_ready_without_npu(self) -> None:
        with patch("ascend_compat.ecosystem.vllm_patch.has_npu", return_value=False):
            result = check_vllm_readiness()
        assert result["ready"] is False
        assert any("NPU" in i for i in result["issues"])

    def test_not_ready_without_vllm(self) -> None:
        with patch("ascend_compat.ecosystem.vllm_patch.has_npu", return_value=True):
            with patch.dict(sys.modules, {"vllm": None}):
                result = check_vllm_readiness()
        assert result["ready"] is False
