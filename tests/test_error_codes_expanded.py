"""Tests for the expanded CANN error code database."""

from __future__ import annotations

import pytest

from ascend_compat.doctor.error_codes import (
    get_all_codes,
    search_errors,
    translate_error,
    format_error,
)


class TestExpandedErrorCodes:
    """Test the newly added error codes."""

    # ── Runtime errors ──────────────────────────────────────────────
    @pytest.mark.parametrize("code", [
        "507001", "507002", "507005", "507014", "507018",
        "507021", "507023",
    ])
    def test_runtime_codes_exist(self, code: str) -> None:
        info = translate_error(code)
        assert info is not None, f"Error code {code} not in database"
        assert info.category == "runtime"

    # ── Operator errors ─────────────────────────────────────────────
    @pytest.mark.parametrize("code", [
        "507030", "507032", "507034", "507036", "507038", "507039",
    ])
    def test_operator_codes_exist(self, code: str) -> None:
        info = translate_error(code)
        assert info is not None, f"Error code {code} not in database"
        assert info.category == "operator"

    # ── Memory errors ───────────────────────────────────────────────
    @pytest.mark.parametrize("code", ["207002", "207003", "207007"])
    def test_memory_codes_exist(self, code: str) -> None:
        info = translate_error(code)
        assert info is not None
        assert info.category == "memory"

    # ── Compile errors ──────────────────────────────────────────────
    @pytest.mark.parametrize("code", ["500003", "500004", "500005"])
    def test_compile_codes_exist(self, code: str) -> None:
        info = translate_error(code)
        assert info is not None
        assert info.category == "compile"

    # ── FP8/Quantization ────────────────────────────────────────────
    def test_fp8_unsupported_code(self) -> None:
        info = translate_error("FP8_UNSUPPORTED")
        assert info is not None
        assert "FP8" in info.summary
        assert "950" in info.likely_cause

    def test_quint8_unsupported_code(self) -> None:
        info = translate_error("QUINT8_UNSUPPORTED")
        assert info is not None
        assert "quint8" in info.summary.lower() or "QUINT8" in info.summary

    # ── Environment errors ──────────────────────────────────────────
    @pytest.mark.parametrize("code", [
        "CANN_NOT_FOUND", "DRIVER_MISMATCH", "FIRMWARE_MISMATCH",
    ])
    def test_environment_codes_exist(self, code: str) -> None:
        info = translate_error(code)
        assert info is not None
        assert info.category == "environment"

    # ── Search functionality ────────────────────────────────────────
    def test_search_fp8(self) -> None:
        results = search_errors("FP8")
        assert len(results) > 0

    def test_search_quantization(self) -> None:
        results = search_errors("quant")
        assert len(results) > 0

    def test_search_hccl(self) -> None:
        results = search_errors("HCCL")
        assert len(results) > 0

    def test_search_tiling(self) -> None:
        results = search_errors("tiling")
        assert len(results) > 0

    # ── Total code count ────────────────────────────────────────────
    def test_at_least_30_codes(self) -> None:
        """We should have a substantial error code database."""
        all_codes = get_all_codes()
        assert len(all_codes) >= 30, f"Only {len(all_codes)} codes in database"

    # ── Format function ─────────────────────────────────────────────
    def test_format_new_code(self) -> None:
        result = format_error("507038")
        assert "Tiling" in result or "tiling" in result

    def test_format_unknown_code(self) -> None:
        result = format_error("999888")
        assert "Unknown" in result
