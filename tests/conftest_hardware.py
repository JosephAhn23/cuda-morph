"""Hardware test configuration.

This module auto-skips ``@pytest.mark.hardware`` tests unless the user
explicitly passes ``--run-hardware`` on the command line.

Usage::

    # Skip hardware tests (default — CI without NPU):
    pytest tests/

    # Run hardware tests (on a machine with Ascend NPU):
    pytest tests/ --run-hardware

    # Run ONLY hardware tests:
    pytest tests/ -m hardware --run-hardware
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``--run-hardware`` CLI flag."""
    parser.addoption(
        "--run-hardware",
        action="store_true",
        default=False,
        help="Run tests that require real Ascend NPU hardware",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-skip hardware tests when --run-hardware is not set."""
    if config.getoption("--run-hardware"):
        return  # User asked for hardware tests — don't skip

    skip_hw = pytest.mark.skip(reason="needs --run-hardware flag and Ascend NPU")
    for item in items:
        if "hardware" in item.keywords:
            item.add_marker(skip_hw)
