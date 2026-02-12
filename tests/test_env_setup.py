"""Tests for the deep environment setup validator."""

from __future__ import annotations

import os
import platform
import sys
from unittest.mock import patch, MagicMock

import pytest

from ascend_compat.doctor.env_setup import (
    COMPAT_MATRIX,
    EnvCheckResult,
    _check_os,
    _check_python,
    _check_cann_installation,
    _check_compilation_tools,
    _check_env_vars,
    _check_disk_space,
    format_env_report,
    full_environment_check,
)


class TestCheckOS:
    """Test OS detection."""

    def test_linux_ubuntu(self) -> None:
        with patch("platform.system", return_value="Linux"):
            with patch("platform.release", return_value="5.4.0"):
                with patch("platform.machine", return_value="x86_64"):
                    with patch(
                        "ascend_compat.doctor.env_setup._get_linux_distro",
                        return_value="Ubuntu 20.04.6 LTS",
                    ):
                        result = _check_os()
        assert result.status == "ok"
        assert "Ubuntu" in result.message

    def test_windows_dev_only(self) -> None:
        with patch("platform.system", return_value="Windows"):
            with patch("platform.release", return_value="10"):
                with patch("platform.machine", return_value="AMD64"):
                    result = _check_os()
        assert result.status == "info"
        assert "development only" in result.message


class TestCheckPython:
    """Test Python version checks."""

    def test_current_python_ok(self) -> None:
        result = _check_python()
        major, minor = sys.version_info.major, sys.version_info.minor
        if major == 3 and 8 <= minor <= 12:
            assert result.status == "ok"

    def test_old_python_error(self) -> None:
        with patch.object(sys, "version_info", MagicMock(major=3, minor=7, micro=0)):
            result = _check_python()
        assert result.status == "error"
        assert "3.8" in result.message


class TestCheckCANNInstallation:
    """Test CANN toolkit checks."""

    def test_missing_cann(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
        with patch("os.path.isdir", return_value=False):
            results = _check_cann_installation()
        assert any(r.status == "error" for r in results)
        assert any("not found" in r.message for r in results)

    def test_found_cann(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")

        def mock_isdir(path: str) -> bool:
            # Simulate that CANN dirs exist
            return True

        def mock_isfile(path: str) -> bool:
            if "version.info" in path:
                return True
            if "libascendcl" in path:
                return True
            return False

        with patch("os.path.isdir", side_effect=mock_isdir):
            with patch("os.path.isfile", side_effect=mock_isfile):
                with patch("builtins.open", MagicMock(
                    return_value=MagicMock(
                        __enter__=lambda s: MagicMock(read=lambda: "8.0.RC3"),
                        __exit__=lambda *a: None,
                    )
                )):
                    results = _check_cann_installation()

        # At least the main CANN check should pass
        assert results[0].status == "ok"
        assert "8.0.RC3" in results[0].message


class TestCheckEnvVars:
    """Test environment variable validation."""

    def test_ascend_home_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ASCEND_HOME_PATH", "/usr/local/Ascend/latest")
        results = _check_env_vars()
        home_result = [r for r in results if "ASCEND_HOME_PATH" in r.name]
        assert len(home_result) == 1
        assert home_result[0].status == "ok"

    def test_ascend_home_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
        results = _check_env_vars()
        home_result = [r for r in results if "ASCEND_HOME_PATH" in r.name]
        assert len(home_result) == 1
        assert home_result[0].status == "warning"


class TestCheckCompilationTools:
    """Test compilation tool checks."""

    def test_cmake_found(self) -> None:
        with patch("shutil.which", return_value="/usr/bin/cmake"):
            import subprocess
            with patch("subprocess.run", return_value=MagicMock(
                stdout="cmake version 3.22.1", returncode=0
            )):
                results = _check_compilation_tools()
        cmake_results = [r for r in results if "cmake" in r.name.lower()]
        assert len(cmake_results) >= 1
        assert cmake_results[0].status == "ok"

    def test_cmake_missing(self) -> None:
        with patch("shutil.which", return_value=None):
            results = _check_compilation_tools()
        cmake_results = [r for r in results if "cmake" in r.name.lower()]
        assert len(cmake_results) >= 1
        assert cmake_results[0].status == "warning"


class TestCheckDiskSpace:
    """Test disk space check."""

    def test_reports_disk_space(self) -> None:
        result = _check_disk_space()
        assert result.status in ("ok", "warning", "info")
        assert "GB" in result.message or "Could not" in result.message


class TestCompatMatrix:
    """Test the compatibility matrix structure."""

    def test_matrix_not_empty(self) -> None:
        assert len(COMPAT_MATRIX) > 0

    def test_matrix_entries_have_required_fields(self) -> None:
        for row in COMPAT_MATRIX:
            assert row.cann
            assert row.driver
            assert row.torch_npu
            assert row.pytorch
            assert row.python_min
            assert row.python_max


class TestFormatReport:
    """Test report formatting."""

    def test_format_with_mixed_results(self) -> None:
        results = [
            EnvCheckResult("Test1", "ok", "All good"),
            EnvCheckResult("Test2", "warning", "Something iffy", fix="Try this"),
            EnvCheckResult("Test3", "error", "Broken", fix="Fix it"),
        ]
        report = format_env_report(results)
        assert "[OK]" in report
        assert "[!!]" in report
        assert "[XX]" in report
        assert "1 passed" in report
        assert "1 warning" in report
        assert "1 error" in report


class TestFullEnvironmentCheck:
    """Integration tests for the full check pipeline."""

    def test_returns_results_list(self) -> None:
        results = full_environment_check()
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, EnvCheckResult) for r in results)
