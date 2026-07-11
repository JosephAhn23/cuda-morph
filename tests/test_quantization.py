"""Tests for the quantization compatibility layer."""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from ascend_compat.cuda_shim.quantization import (
    QuantCompat,
    _detect_quant_method,
    check_model_quant,
    check_quant_method,
    format_quant_report,
    get_supported_methods,
    get_unsupported_methods,
)


class TestCheckQuantMethod:
    """Test quantization method compatibility checks."""

    def test_fp8_unsupported(self) -> None:
        result = check_quant_method("fp8")
        assert not result.supported
        assert result.alternative == "w8a8_dynamic"
        assert "950" in result.suggestion

    def test_fp8_e4m3_unsupported(self) -> None:
        result = check_quant_method("fp8_e4m3")
        assert not result.supported

    def test_awq_unsupported(self) -> None:
        result = check_quant_method("awq")
        assert not result.supported
        assert result.alternative == "w8a8"

    def test_gptq_unsupported(self) -> None:
        result = check_quant_method("gptq")
        assert not result.supported

    def test_bitsandbytes_unsupported(self) -> None:
        result = check_quant_method("bitsandbytes")
        assert not result.supported

    def test_marlin_unsupported(self) -> None:
        result = check_quant_method("marlin")
        assert not result.supported

    def test_w8a8_supported(self) -> None:
        result = check_quant_method("w8a8")
        assert result.supported
        assert result.native_performance

    def test_w8a8_dynamic_supported(self) -> None:
        result = check_quant_method("w8a8_dynamic")
        assert result.supported

    def test_smoothquant_supported(self) -> None:
        result = check_quant_method("smoothquant")
        assert result.supported

    def test_none_supported(self) -> None:
        result = check_quant_method("none")
        assert result.supported

    def test_empty_string_is_none(self) -> None:
        result = check_quant_method("")
        assert result.supported  # maps to "none"

    def test_unknown_method(self) -> None:
        result = check_quant_method("future_quant_xyz")
        assert not result.supported
        assert "not recognized" in result.suggestion

    def test_case_insensitive(self) -> None:
        result = check_quant_method("FP8")
        assert not result.supported


class TestDetectQuantMethod:
    """Test model quantization method detection."""

    def test_detects_from_config_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"quantization_config": {"quant_method": "gptq"}}
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump(config, f)
            result = _detect_quant_method(tmpdir)
            assert result == "gptq"

    def test_detects_from_model_name_gptq(self) -> None:
        result = _detect_quant_method("TheBloke/Llama-2-7B-GPTQ")
        assert result == "gptq"

    def test_detects_from_model_name_awq(self) -> None:
        result = _detect_quant_method("TheBloke/Llama-2-7B-AWQ")
        assert result == "awq"

    def test_detects_from_model_name_fp8(self) -> None:
        result = _detect_quant_method("neuralmagic/Llama-3-8B-FP8")
        assert result == "fp8"

    def test_defaults_to_none(self) -> None:
        result = _detect_quant_method("meta-llama/Llama-3-8B")
        assert result == "none"


class TestCheckModelQuant:
    """Test end-to-end model quantization check."""

    def test_check_gptq_model(self) -> None:
        result = check_model_quant("TheBloke/Llama-2-7B-GPTQ")
        assert not result.supported
        assert result.method == "gptq"

    def test_check_vanilla_model(self) -> None:
        result = check_model_quant("meta-llama/Llama-3-8B")
        assert result.supported


class TestGetMethods:
    """Test method listing functions."""

    def test_supported_methods_not_empty(self) -> None:
        methods = get_supported_methods()
        assert len(methods) > 0
        assert "w8a8" in methods

    def test_unsupported_methods_not_empty(self) -> None:
        methods = get_unsupported_methods()
        assert len(methods) > 0
        assert "fp8" in methods


class TestFormatReport:
    """Test report formatting."""

    def test_format_supported(self) -> None:
        compat = QuantCompat("w8a8", True, True, "Supported")
        report = format_quant_report(compat)
        assert "[OK]" in report

    def test_format_unsupported(self) -> None:
        compat = QuantCompat("fp8", False, False, "Not supported", "w8a8")
        report = format_quant_report(compat)
        assert "[XX]" in report
        assert "Recommended: w8a8" in report
