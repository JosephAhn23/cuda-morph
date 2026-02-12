"""Tests for the CLI tool (cli.py).

Tests verify:
1. File scanning (check) finds CUDA dependencies correctly
2. Auto-porting (port) rewrites simple patterns
3. Migration difficulty assessment
4. Edge cases (empty files, syntax errors, no CUDA usage)
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from ascend_compat.cli import CheckReport, check_file, main, port_file, show_info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_temp_py(content: str) -> str:
    """Write content to a temporary .py file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".py", prefix="ascend_test_")
    os.write(fd, content.encode("utf-8"))
    os.close(fd)
    return path


# ---------------------------------------------------------------------------
# check_file tests
# ---------------------------------------------------------------------------


class TestCheckFile:
    """Test the check_file() scanner."""

    def test_empty_file(self) -> None:
        path = _write_temp_py("")
        try:
            report = check_file(path)
            assert report.total_cuda_refs == 0
            assert report.migration_difficulty == "trivial"
        finally:
            os.unlink(path)

    def test_no_cuda_usage(self) -> None:
        code = """
import torch
x = torch.randn(3, 3)
y = x + 1
"""
        path = _write_temp_py(code)
        try:
            report = check_file(path)
            assert report.total_cuda_refs == 0
            assert report.migration_difficulty == "trivial"
        finally:
            os.unlink(path)

    def test_detects_cuda_is_available(self) -> None:
        code = """
import torch
if torch.cuda.is_available():
    device = "cuda"
"""
        path = _write_temp_py(code)
        try:
            report = check_file(path)
            assert report.total_cuda_refs > 0
            api_calls = [d.api_call for d in report.dependencies]
            assert any("is_available" in c for c in api_calls)
        finally:
            os.unlink(path)

    def test_detects_dot_cuda_call(self) -> None:
        code = """
import torch
x = torch.randn(3, 3)
x = x.cuda()
"""
        path = _write_temp_py(code)
        try:
            report = check_file(path)
            assert any(d.api_call == ".cuda()" for d in report.dependencies)
        finally:
            os.unlink(path)

    def test_detects_cuda_device_strings(self) -> None:
        code = """
import torch
device = "cuda"
x = torch.randn(3, 3, device="cuda:0")
"""
        path = _write_temp_py(code)
        try:
            report = check_file(path)
            assert report.has_cuda_device_strings
        finally:
            os.unlink(path)

    def test_detects_amp_autocast(self) -> None:
        code = """
import torch
from torch.cuda.amp import autocast, GradScaler

with autocast():
    pass
"""
        path = _write_temp_py(code)
        try:
            report = check_file(path)
            assert report.imports_torch_cuda
        finally:
            os.unlink(path)

    def test_detects_cudnn(self) -> None:
        code = """
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
"""
        path = _write_temp_py(code)
        try:
            report = check_file(path)
            assert any("cudnn" in d.api_call for d in report.dependencies)
        finally:
            os.unlink(path)

    def test_detects_manual_seed(self) -> None:
        code = """
import torch
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
"""
        path = _write_temp_py(code)
        try:
            report = check_file(path)
            api_calls = [d.api_call for d in report.dependencies]
            assert any("manual_seed" in c for c in api_calls)
        finally:
            os.unlink(path)

    def test_detects_unsupported_cuda_graph(self) -> None:
        code = """
import torch
g = torch.cuda.CUDAGraph()
"""
        path = _write_temp_py(code)
        try:
            report = check_file(path)
            assert any(d.status == "unsupported" for d in report.dependencies)
            assert report.migration_difficulty == "hard"
        finally:
            os.unlink(path)

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            check_file("/nonexistent/path/model.py")

    def test_complex_training_script(self) -> None:
        """Test a realistic training script with multiple CUDA patterns."""
        code = """
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

device = "cuda" if torch.cuda.is_available() else "cpu"
model = nn.Linear(10, 5).cuda()
optimizer = torch.optim.Adam(model.parameters())
scaler = GradScaler()

torch.cuda.manual_seed(42)
torch.backends.cudnn.benchmark = True

for epoch in range(10):
    x = torch.randn(32, 10, device="cuda")
    with autocast():
        y = model(x)
        loss = y.sum()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

print(f"Memory: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
torch.cuda.empty_cache()
"""
        path = _write_temp_py(code)
        try:
            report = check_file(path)
            assert report.total_cuda_refs > 5
            assert report.imports_torch_cuda
            # Should find: is_available, .cuda(), manual_seed, cudnn.benchmark,
            # memory_allocated, empty_cache, device strings
        finally:
            os.unlink(path)


class TestMigrationDifficulty:
    """Test the migration difficulty assessment."""

    def test_trivial(self) -> None:
        path = _write_temp_py("x = 1")
        try:
            report = check_file(path)
            assert report.migration_difficulty == "trivial"
        finally:
            os.unlink(path)

    def test_easy(self) -> None:
        code = """
import torch
torch.cuda.empty_cache()
"""
        path = _write_temp_py(code)
        try:
            report = check_file(path)
            # All dependencies are "needs_wrapper" with no unknowns → easy
            assert report.migration_difficulty == "easy"
        finally:
            os.unlink(path)

    def test_hard_with_unsupported(self) -> None:
        code = """
import torch
g = torch.cuda.CUDAGraph()
"""
        path = _write_temp_py(code)
        try:
            report = check_file(path)
            assert report.migration_difficulty == "hard"
        finally:
            os.unlink(path)


class TestReportSummary:
    """Test that the report summary renders correctly."""

    def test_summary_is_string(self) -> None:
        code = """
import torch
torch.cuda.is_available()
"""
        path = _write_temp_py(code)
        try:
            report = check_file(path)
            summary = report.summary()
            assert isinstance(summary, str)
            assert "ascend-compat" in summary
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# port_file tests
# ---------------------------------------------------------------------------


class TestPortFile:
    """Test the port_file() auto-rewriter."""

    def test_adds_import(self) -> None:
        code = """import torch
x = torch.randn(3)
"""
        path = _write_temp_py(code)
        try:
            result = port_file(path, dry_run=True)
            assert "import ascend_compat" in result
        finally:
            os.unlink(path)

    def test_replaces_is_available(self) -> None:
        code = """import torch
if torch.cuda.is_available():
    pass
"""
        path = _write_temp_py(code)
        try:
            result = port_file(path, dry_run=True)
            assert "ascend_compat.device.is_available()" in result
            assert "torch.cuda.is_available()" not in result
        finally:
            os.unlink(path)

    def test_replaces_empty_cache(self) -> None:
        code = """import torch
torch.cuda.empty_cache()
"""
        path = _write_temp_py(code)
        try:
            result = port_file(path, dry_run=True)
            assert "ascend_compat.memory.empty_cache()" in result
        finally:
            os.unlink(path)

    def test_replaces_manual_seed(self) -> None:
        code = """import torch
torch.cuda.manual_seed(42)
"""
        path = _write_temp_py(code)
        try:
            result = port_file(path, dry_run=True)
            assert "ascend_compat.ops.manual_seed(42)" in result
        finally:
            os.unlink(path)

    def test_dry_run_does_not_write(self) -> None:
        code = """import torch
torch.cuda.is_available()
"""
        path = _write_temp_py(code)
        try:
            original = Path(path).read_text()
            port_file(path, dry_run=True)
            # File should be unchanged
            assert Path(path).read_text() == original
        finally:
            os.unlink(path)

    def test_creates_backup(self) -> None:
        code = """import torch
torch.cuda.is_available()
"""
        path = _write_temp_py(code)
        try:
            port_file(path, dry_run=False)
            backup = Path(path + ".bak")
            assert backup.exists()
            assert backup.read_text() == code
            backup.unlink()
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# show_info tests
# ---------------------------------------------------------------------------


class TestShowInfo:
    """Test the info command."""

    def test_returns_string(self) -> None:
        info = show_info()
        assert isinstance(info, str)
        assert "PyTorch" in info


# ---------------------------------------------------------------------------
# CLI main() tests
# ---------------------------------------------------------------------------


class TestCLIMain:
    """Test the CLI entry point."""

    def test_no_args_shows_help(self, capsys: pytest.CaptureFixture) -> None:
        result = main([])
        assert result == 0

    def test_info_command(self, capsys: pytest.CaptureFixture) -> None:
        result = main(["info"])
        assert result == 0
        captured = capsys.readouterr()
        assert "PyTorch" in captured.out

    def test_check_command(self, capsys: pytest.CaptureFixture) -> None:
        code = """import torch
torch.cuda.is_available()
"""
        path = _write_temp_py(code)
        try:
            result = main(["check", path])
            assert result == 0
            captured = capsys.readouterr()
            assert "ascend-compat" in captured.out
        finally:
            os.unlink(path)

    def test_check_json_output(self, capsys: pytest.CaptureFixture) -> None:
        code = """import torch
torch.cuda.is_available()
"""
        path = _write_temp_py(code)
        try:
            result = main(["check", path, "--json"])
            assert result == 0
            captured = capsys.readouterr()
            data = json.loads(captured.out)
            assert "dependencies" in data
            assert "migration_difficulty" in data
        finally:
            os.unlink(path)

    def test_check_nonexistent_file(self) -> None:
        result = main(["check", "/nonexistent/file.py"])
        assert result == 1

    def test_port_dry_run(self, capsys: pytest.CaptureFixture) -> None:
        code = """import torch
torch.cuda.is_available()
"""
        path = _write_temp_py(code)
        try:
            result = main(["port", path, "--dry-run"])
            assert result == 0
        finally:
            os.unlink(path)

    def test_doctor_command(self, capsys: pytest.CaptureFixture) -> None:
        result = main(["doctor"])
        assert result == 0
        captured = capsys.readouterr()
        assert "ascend-compat doctor" in captured.out

    def test_error_command_known_code(self, capsys: pytest.CaptureFixture) -> None:
        result = main(["error", "507035"])
        assert result == 0
        captured = capsys.readouterr()
        assert "507035" in captured.out

    def test_error_command_unknown_code(self, capsys: pytest.CaptureFixture) -> None:
        result = main(["error", "999999"])
        assert result == 0
        captured = capsys.readouterr()
        assert "Unknown" in captured.out
