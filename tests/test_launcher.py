"""Tests for the launcher (ascend-compat run, python -m ascend_compat)."""

from __future__ import annotations

import os
import tempfile

import pytest

from ascend_compat.cli import main


class TestRunCommand:
    """Tests for the `ascend-compat run` command."""

    def test_run_simple_script(self) -> None:
        """Run a simple Python script through the launcher."""
        code = "import sys; print('hello from ascend-compat')"
        fd, path = tempfile.mkstemp(suffix=".py")
        os.write(fd, code.encode())
        os.close(fd)

        try:
            result = main(["run", path])
            assert result == 0
        finally:
            os.unlink(path)

    def test_run_script_with_args(self, capsys: pytest.CaptureFixture) -> None:
        """Script args should be passed through."""
        code = "import sys; print(','.join(sys.argv[1:]))"
        fd, path = tempfile.mkstemp(suffix=".py")
        os.write(fd, code.encode())
        os.close(fd)

        try:
            result = main(["run", path, "--batch-size", "32"])
            assert result == 0
            captured = capsys.readouterr()
            assert "--batch-size,32" in captured.out
        finally:
            os.unlink(path)

    def test_run_nonexistent_script(self) -> None:
        """Running a nonexistent script should return error."""
        result = main(["run", "/nonexistent/script.py"])
        assert result == 1

    def test_run_script_that_imports_torch(self) -> None:
        """Script that uses torch should work."""
        code = "import torch; print(f'torch={torch.__version__}')"
        fd, path = tempfile.mkstemp(suffix=".py")
        os.write(fd, code.encode())
        os.close(fd)

        try:
            result = main(["run", path])
            assert result == 0
        finally:
            os.unlink(path)

    def test_run_script_with_exit_code(self) -> None:
        """Script that calls sys.exit should propagate the code."""
        code = "import sys; sys.exit(42)"
        fd, path = tempfile.mkstemp(suffix=".py")
        os.write(fd, code.encode())
        os.close(fd)

        try:
            result = main(["run", path])
            assert result == 42
        finally:
            os.unlink(path)


class TestPythonMEntry:
    """Test that python -m ascend_compat works."""

    def test_main_module_exists(self) -> None:
        """The __main__.py module should be importable."""
        import ascend_compat.__main__
        assert hasattr(ascend_compat.__main__, "main")
