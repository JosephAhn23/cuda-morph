"""Tests for the doctor module (version_check, error_codes, fallback_monitor)."""

from __future__ import annotations

import warnings

from ascend_compat.doctor.version_check import (
    CheckResult,
    check_versions,
    format_report,
)
from ascend_compat.doctor.error_codes import (
    format_error,
    get_all_codes,
    search_errors,
    translate_error,
)
from ascend_compat.doctor.fallback_monitor import (
    FallbackMonitor,
    FallbackReport,
    _extract_op_name,
)


class TestVersionCheck:
    """Tests for version compatibility checking."""

    def test_check_versions_returns_list(self) -> None:
        results = check_versions()
        assert isinstance(results, list)
        assert len(results) > 0

    def test_python_check_present(self) -> None:
        results = check_versions()
        names = [r.name for r in results]
        assert "Python" in names

    def test_pytorch_check_present(self) -> None:
        results = check_versions()
        names = [r.name for r in results]
        assert "PyTorch" in names

    def test_format_report_produces_string(self) -> None:
        results = check_versions()
        report = format_report(results)
        assert isinstance(report, str)
        assert "ascend-compat doctor" in report

    def test_check_result_statuses(self) -> None:
        """All statuses should be valid."""
        results = check_versions()
        for r in results:
            assert r.status in ("ok", "warning", "error", "skipped")


class TestErrorCodes:
    """Tests for the CANN error code translator."""

    def test_known_code_507035(self) -> None:
        info = translate_error(507035)
        assert info is not None
        assert info.code == "507035"
        assert info.category == "operator"
        assert "kernel" in info.summary.lower()

    def test_known_code_string(self) -> None:
        info = translate_error("507008")
        assert info is not None
        assert info.category == "runtime"

    def test_err99999(self) -> None:
        info = translate_error("ERR99999")
        assert info is not None
        assert "UNKNOWN" in info.summary

    def test_unknown_code_returns_none(self) -> None:
        assert translate_error(999888777) is None

    def test_format_error_known(self) -> None:
        output = format_error(507035)
        assert "507035" in output
        assert "operator" in output.lower() or "Category" in output

    def test_format_error_unknown(self) -> None:
        output = format_error(999888777)
        assert "Unknown" in output

    def test_get_all_codes_nonempty(self) -> None:
        codes = get_all_codes()
        assert len(codes) > 10

    def test_search_errors_memory(self) -> None:
        results = search_errors("memory")
        assert len(results) > 0
        # At least the memory allocation error should match
        codes = [r.code for r in results]
        assert any("207" in c or "507011" in c for c in codes)

    def test_search_errors_case_insensitive(self) -> None:
        r1 = search_errors("MEMORY")
        r2 = search_errors("memory")
        assert len(r1) == len(r2)


class TestFallbackMonitor:
    """Tests for the CPU fallback monitor."""

    def test_monitor_context_manager(self) -> None:
        """Monitor should work as a context manager."""
        with FallbackMonitor() as monitor:
            pass  # No actual NPU ops
        assert isinstance(monitor.report, FallbackReport)
        assert monitor.report.total_fallbacks == 0

    def test_monitor_start_stop(self) -> None:
        monitor = FallbackMonitor()
        monitor.start()
        report = monitor.stop()
        assert report.total_fallbacks == 0
        assert report.monitoring_duration_s >= 0

    def test_monitor_detects_fallback_warning(self) -> None:
        """Monitor should detect fallback warnings."""
        with FallbackMonitor() as monitor:
            # Simulate the warning torch_npu would emit
            warnings.warn(
                "The operator 'aten::_unique2' is not currently supported "
                "on the NPU backend and will fall back to run on the CPU",
                UserWarning,
                stacklevel=1,
            )

        assert monitor.report.total_fallbacks == 1
        assert "aten::_unique2" in monitor.report.stats

    def test_monitor_counts_multiple(self) -> None:
        import importlib

        with FallbackMonitor() as monitor:
            for i in range(5):
                # Each warning needs unique message or location to avoid
                # Python's warning deduplication filter
                warnings.warn(
                    f"The operator 'aten::histc' will fall back to CPU (call {i})",
                    UserWarning,
                    stacklevel=1,
                )

        assert monitor.report.total_fallbacks == 5
        assert monitor.report.stats["aten::histc"].call_count == 5

    def test_report_summary_no_fallbacks(self) -> None:
        report = FallbackReport()
        summary = report.summary()
        assert "No CPU fallbacks" in summary

    def test_report_summary_with_fallbacks(self) -> None:
        from ascend_compat.doctor.fallback_monitor import FallbackStats
        report = FallbackReport(
            total_fallbacks=10,
            unique_ops=2,
            monitoring_duration_s=5.0,
            stats={
                "aten::histc": FallbackStats(
                    op_name="aten::histc", call_count=7
                ),
                "aten::_unique2": FallbackStats(
                    op_name="aten::_unique2", call_count=3
                ),
            },
        )
        summary = report.summary()
        assert "10" in summary
        assert "aten::histc" in summary
        assert "aten::_unique2" in summary

    def test_extract_op_name_quoted(self) -> None:
        msg = "The operator 'aten::_unique2' is not currently supported"
        assert _extract_op_name(msg) == "aten::_unique2"

    def test_extract_op_name_unquoted(self) -> None:
        msg = "operator aten::histc will fall back to CPU"
        assert _extract_op_name(msg) == "aten::histc"

    def test_extract_op_name_unknown_format(self) -> None:
        msg = "Something weird happened with CPU fallback"
        result = _extract_op_name(msg)
        assert isinstance(result, str)
        assert len(result) > 0
