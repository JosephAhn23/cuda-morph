"""Tests for the import hook (cuda_shim._import_hook)."""

from __future__ import annotations

import sys

from ascend_compat.cuda_shim._import_hook import (
    _AscendCompatFinder,
    install_import_hook,
    uninstall_import_hook,
)


class TestImportHook:
    """Tests for the sys.meta_path import hook."""

    def test_install_adds_finder(self) -> None:
        """install_import_hook should add a finder to sys.meta_path."""
        # Clean up first
        uninstall_import_hook()
        initial_count = len(sys.meta_path)

        install_import_hook()
        assert len(sys.meta_path) == initial_count + 1
        assert any(isinstance(f, _AscendCompatFinder) for f in sys.meta_path)

        uninstall_import_hook()

    def test_install_is_idempotent(self) -> None:
        """Multiple install calls should not duplicate the finder."""
        uninstall_import_hook()
        install_import_hook()
        install_import_hook()  # Second call should be no-op

        finder_count = sum(1 for f in sys.meta_path if isinstance(f, _AscendCompatFinder))
        assert finder_count == 1

        uninstall_import_hook()

    def test_uninstall_removes_finder(self) -> None:
        """uninstall_import_hook should remove all our finders."""
        install_import_hook()
        uninstall_import_hook()
        assert not any(isinstance(f, _AscendCompatFinder) for f in sys.meta_path)

    def test_finder_only_matches_torch_cuda(self) -> None:
        """The finder should only intercept torch.cuda imports."""
        finder = _AscendCompatFinder()
        assert finder.find_module("torch.cuda") is not None
        assert finder.find_module("torch.cuda.amp") is not None
        assert finder.find_module("torch.nn") is None
        assert finder.find_module("numpy") is None
        assert finder.find_module("torch") is None  # Only torch.cuda, not torch itself
