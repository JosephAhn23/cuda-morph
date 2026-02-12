"""Structured logging for the ascend-compat shim layer.

Every translation from a CUDA API call to an Ascend/CANN equivalent is logged
at DEBUG level so developers can see exactly what the shim is doing under the
hood.  This is critical for debugging: when something behaves differently on
Ascend vs CUDA, the logs tell you which mapping was used.

Usage:
    from ascend_compat._logging import get_logger
    logger = get_logger(__name__)
    logger.debug("torch.cuda.is_available() → torch_npu.npu.is_available()")
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_LOG_FORMAT = "[ascend-compat] %(levelname)s %(name)s: %(message)s"
_LOG_FORMAT_VERBOSE = (
    "[ascend-compat %(asctime)s] %(levelname)s %(name)s (%(filename)s:%(lineno)d): %(message)s"
)

# Environment variable that users can set to control verbosity.
# Values: DEBUG, INFO, WARNING, ERROR, CRITICAL  (case-insensitive)
_ENV_LOG_LEVEL = "ASCEND_COMPAT_LOG_LEVEL"

# When set to "1", use the verbose format that includes timestamps and line numbers.
_ENV_LOG_VERBOSE = "ASCEND_COMPAT_LOG_VERBOSE"


# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

_root_logger_configured = False


def _resolve_log_level() -> int:
    """Read the desired log level from the environment, defaulting to WARNING.

    Why WARNING as default?
    -----------------------
    During normal operation the shim should be silent.  Developers who are
    *debugging* a migration set ``ASCEND_COMPAT_LOG_LEVEL=DEBUG`` to see every
    single API mapping.  This keeps the console clean for end-users while
    giving full transparency when needed.
    """
    env_val = os.environ.get(_ENV_LOG_LEVEL, "").strip().upper()
    return getattr(logging, env_val, logging.WARNING)


def _configure_root_logger() -> None:
    """One-time setup of the ``ascend_compat`` logger hierarchy.

    We attach a :class:`logging.StreamHandler` (stderr) to the top-level
    ``ascend_compat`` logger so that every sub-logger (e.g.
    ``ascend_compat.device``) inherits the handler automatically.
    """
    global _root_logger_configured  # noqa: PLW0603
    if _root_logger_configured:
        return

    root = logging.getLogger("ascend_compat")
    root.setLevel(_resolve_log_level())

    # Only add our handler if one hasn't already been added (e.g. by a test
    # fixture or the user's own logging config).
    if not root.handlers:
        verbose = os.environ.get(_ENV_LOG_VERBOSE, "0").strip() == "1"
        fmt = _LOG_FORMAT_VERBOSE if verbose else _LOG_FORMAT
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(fmt))
        root.addHandler(handler)

    _root_logger_configured = True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_logger(name: str) -> logging.Logger:
    """Return a logger scoped under the ``ascend_compat`` namespace.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A :class:`logging.Logger` that inherits the shim's root handler.
    """
    _configure_root_logger()
    return logging.getLogger(name)


def set_log_level(level: Optional[str] = None) -> None:
    """Programmatically change the shim's log level at runtime.

    This is a convenience for interactive sessions (Jupyter notebooks, REPL)
    where editing environment variables is annoying.

    Args:
        level: One of ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``,
            ``"CRITICAL"``.  If *None*, resets to the environment default.

    Example::

        import ascend_compat
        ascend_compat.set_log_level("DEBUG")  # see every mapping
    """
    _configure_root_logger()
    root = logging.getLogger("ascend_compat")
    if level is None:
        root.setLevel(_resolve_log_level())
    else:
        root.setLevel(getattr(logging, level.upper(), logging.WARNING))
