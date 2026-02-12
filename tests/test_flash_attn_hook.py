"""Tests for the flash_attn import hook."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from ascend_compat.ecosystem._flash_attn_hook import (
    _FlashAttnFinder,
    install_flash_attn_hook,
    uninstall_flash_attn_hook,
)


class TestFlashAttnHook:
    """Tests for the flash_attn import hook."""

    def setup_method(self) -> None:
        """Clean up flash_attn from sys.modules before each test."""
        for key in list(sys.modules):
            if key.startswith("flash_attn") and "ascend_compat" not in key:
                del sys.modules[key]
        uninstall_flash_attn_hook()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        uninstall_flash_attn_hook()
        for key in list(sys.modules):
            if key == "flash_attn" or (key.startswith("flash_attn.") and "ascend_compat" not in key):
                del sys.modules[key]

    def test_install_adds_finder(self) -> None:
        install_flash_attn_hook()
        assert any(isinstance(f, _FlashAttnFinder) for f in sys.meta_path)

    def test_install_is_idempotent(self) -> None:
        install_flash_attn_hook()
        install_flash_attn_hook()
        count = sum(1 for f in sys.meta_path if isinstance(f, _FlashAttnFinder))
        assert count == 1

    def test_uninstall_removes_finder(self) -> None:
        install_flash_attn_hook()
        uninstall_flash_attn_hook()
        assert not any(isinstance(f, _FlashAttnFinder) for f in sys.meta_path)

    def test_hook_serves_shim_module(self) -> None:
        """After installing the hook, `import flash_attn` should return our shim."""
        install_flash_attn_hook()

        # Force fresh import
        for key in list(sys.modules):
            if key.startswith("flash_attn") and "ascend_compat" not in key:
                del sys.modules[key]

        import importlib
        fa = importlib.import_module("flash_attn")

        # Should have our function
        assert hasattr(fa, "flash_attn_func")
        assert hasattr(fa, "flash_attn_varlen_func")
        assert hasattr(fa, "flash_attn_with_kvcache")

    def test_hook_serves_submodule(self) -> None:
        """flash_attn.flash_attn_interface should also resolve to our shim."""
        install_flash_attn_hook()

        for key in list(sys.modules):
            if key.startswith("flash_attn") and "ascend_compat" not in key:
                del sys.modules[key]

        import importlib
        fai = importlib.import_module("flash_attn.flash_attn_interface")
        assert hasattr(fai, "flash_attn_func")

    def test_finder_only_matches_flash_attn(self) -> None:
        finder = _FlashAttnFinder()
        assert finder.find_module("flash_attn") is not None
        assert finder.find_module("flash_attn.flash_attn_interface") is not None
        assert finder.find_module("torch") is None
        assert finder.find_module("numpy") is None
