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

from click.testing import CliRunner

from ascend_compat.cli import CheckReport, check_file, main, port_file, show_info

_runner = CliRunner()


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

    def test_adds_activate(self) -> None:
        """port_file now adds shim activation instead of rewriting individual calls."""
        code = """import torch
if torch.cuda.is_available():
    pass
"""
        path = _write_temp_py(code)
        try:
            result = port_file(path, dry_run=True)
            assert "import ascend_compat" in result
            assert "ascend_compat.activate()" in result
            # Original calls are preserved (shim handles them at runtime)
            assert "torch.cuda.is_available()" in result
        finally:
            os.unlink(path)

    def test_preserves_cuda_calls(self) -> None:
        """Verify that port_file does NOT rewrite individual torch.cuda calls."""
        code = """import torch
torch.cuda.empty_cache()
"""
        path = _write_temp_py(code)
        try:
            result = port_file(path, dry_run=True)
            assert "ascend_compat.activate()" in result
            # Call is preserved — the shim redirects it at runtime
            assert "torch.cuda.empty_cache()" in result
        finally:
            os.unlink(path)

    def test_idempotent(self) -> None:
        """Porting a file that already has activate() is a no-op."""
        code = """import torch
import ascend_compat
ascend_compat.activate()
torch.cuda.manual_seed(42)
"""
        path = _write_temp_py(code)
        try:
            result = port_file(path, dry_run=True)
            # Should not double-add
            assert result.count("ascend_compat.activate()") == 1
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
    """Test the CLI entry point (Click-based)."""

    def test_no_args_shows_help(self) -> None:
        result = _runner.invoke(main, [])
        # Click groups return exit_code 0 for --help but 2 for no-args
        assert result.exit_code in (0, 2)
        assert "cuda-morph" in result.output

    def test_info_command(self) -> None:
        result = _runner.invoke(main, ["info"])
        assert result.exit_code == 0
        assert "PyTorch" in result.output

    def test_check_command(self) -> None:
        code = """import torch
torch.cuda.is_available()
"""
        path = _write_temp_py(code)
        try:
            result = _runner.invoke(main, ["check", path])
            assert result.exit_code == 0
            assert "ascend-compat" in result.output
        finally:
            os.unlink(path)

    def test_check_json_output(self) -> None:
        code = """import torch
torch.cuda.is_available()
"""
        path = _write_temp_py(code)
        try:
            result = _runner.invoke(main, ["check", path, "--json"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert "dependencies" in data
            assert "migration_difficulty" in data
        finally:
            os.unlink(path)

    def test_check_nonexistent_file(self) -> None:
        result = _runner.invoke(main, ["check", "/nonexistent/file.py"])
        assert result.exit_code != 0

    def test_port_dry_run(self) -> None:
        code = """import torch
torch.cuda.is_available()
"""
        path = _write_temp_py(code)
        try:
            result = _runner.invoke(main, ["port", path, "--dry-run"])
            assert result.exit_code == 0
        finally:
            os.unlink(path)

    def test_doctor_command(self) -> None:
        result = _runner.invoke(main, ["doctor"])
        assert result.exit_code == 0
        assert "ascend-compat doctor" in result.output

    def test_error_command_known_code(self) -> None:
        result = _runner.invoke(main, ["error", "507035"])
        assert result.exit_code == 0
        assert "507035" in result.output

    def test_error_command_unknown_code(self) -> None:
        result = _runner.invoke(main, ["error", "999999"])
        assert result.exit_code == 0
        assert "Unknown" in result.output
