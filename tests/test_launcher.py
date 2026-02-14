"""Tests for the launcher (ascend-compat run, python -m ascend_compat)."""

from __future__ import annotations

import os
import tempfile

import pytest
from click.testing import CliRunner

from ascend_compat.cli import main

_runner = CliRunner()


class TestRunCommand:
    """Tests for the `ascend-compat run` command."""

    def test_run_simple_script(self) -> None:
        """Run a simple Python script through the launcher."""
        code = "import sys; print('hello from ascend-compat')"
        fd, path = tempfile.mkstemp(suffix=".py")
        os.write(fd, code.encode())
        os.close(fd)

        try:
            result = _runner.invoke(main, ["run", path])
            assert result.exit_code == 0
        finally:
            os.unlink(path)

    def test_run_script_with_args(self) -> None:
        """Script args should be passed through."""
        code = "import sys; print(','.join(sys.argv[1:]))"
        fd, path = tempfile.mkstemp(suffix=".py")
        os.write(fd, code.encode())
        os.close(fd)

        try:
            result = _runner.invoke(main, ["run", path, "--", "--batch-size", "32"])
            assert result.exit_code == 0
            assert "--batch-size,32" in result.output
        finally:
            os.unlink(path)

    def test_run_nonexistent_script(self) -> None:
        """Running a nonexistent script should return error."""
        result = _runner.invoke(main, ["run", "/nonexistent/script.py"])
        assert result.exit_code != 0

    def test_run_script_that_imports_torch(self) -> None:
        """Script that uses torch should work."""
        code = "import torch; print(f'torch={torch.__version__}')"
        fd, path = tempfile.mkstemp(suffix=".py")
        os.write(fd, code.encode())
        os.close(fd)

        try:
            result = _runner.invoke(main, ["run", path])
            assert result.exit_code == 0
        finally:
            os.unlink(path)


class TestPythonMEntry:
    """Test that python -m ascend_compat works."""

    def test_main_module_exists(self) -> None:
        """The __main__.py module should be importable."""
        import ascend_compat.__main__
        assert hasattr(ascend_compat.__main__, "main")
