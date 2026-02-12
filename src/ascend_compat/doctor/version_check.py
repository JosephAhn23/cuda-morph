"""Version compatibility verification for the Ascend software stack.

The Ascend ecosystem has very strict version coupling:
  CANN version ↔ torch_npu version ↔ PyTorch version ↔ driver/firmware

Mismatches produce cryptic errors.  This module validates the installed
versions against a known-good compatibility matrix.

Compatibility Matrix (sourced from Huawei's official documentation)
-------------------------------------------------------------------

+----------------+--------------+----------------+-------------------+
| CANN Version   | torch_npu    | PyTorch        | Python            |
+================+==============+================+===================+
| 8.0.RC3        | 2.5.1        | 2.5.1          | 3.8-3.12          |
| 8.0.RC2        | 2.4.0        | 2.4.0          | 3.8-3.12          |
| 8.0.RC1        | 2.3.1        | 2.3.1          | 3.8-3.11          |
| 7.0.1          | 2.2.0        | 2.2.0          | 3.8-3.11          |
| 7.0.0          | 2.1.0        | 2.1.0          | 3.8-3.10          |
+----------------+--------------+----------------+-------------------+

Note: This matrix should be updated as new versions are released.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ascend_compat._logging import get_logger

logger = get_logger(__name__)


@dataclass
class VersionInfo:
    """Detected component versions."""
    python: str = ""
    pytorch: str = ""
    torch_npu: str = ""
    cann: str = ""
    driver: str = ""
    os: str = ""


@dataclass
class CheckResult:
    """Result of a single compatibility check."""
    name: str
    status: str  # "ok", "warning", "error", "skipped"
    message: str
    detail: str = ""


# Known-good version combinations: (torch_npu_prefix, pytorch_prefix, cann_prefix)
_COMPAT_MATRIX: List[Tuple[str, str, str]] = [
    ("2.5", "2.5", "8.0"),
    ("2.4", "2.4", "8.0"),
    ("2.3", "2.3", "8.0"),
    ("2.2", "2.2", "7.0"),
    ("2.1", "2.1", "7.0"),
    ("2.0", "2.0", "6.3"),
]


def check_versions() -> List[CheckResult]:
    """Run all version compatibility checks.

    Returns:
        List of :class:`CheckResult` items.
    """
    results: List[CheckResult] = []
    info = _detect_versions()

    # 1. Python version
    results.append(_check_python(info))

    # 2. PyTorch version
    results.append(_check_pytorch(info))

    # 3. torch_npu
    results.append(_check_torch_npu(info))

    # 4. CANN
    results.append(_check_cann(info))

    # 5. Cross-version compatibility
    results.append(_check_cross_compat(info))

    # 6. NPU device availability
    results.append(_check_npu_device(info))

    return results


def _detect_versions() -> VersionInfo:
    """Detect all component versions.

    Uses ``ascend_compat._backend`` for PyTorch and torch_npu imports to
    avoid duplicating lazy-import logic.  CANN version detection is specific
    to this module since ``_backend`` doesn't track it.
    """
    from ascend_compat._backend import get_torch, get_torch_npu

    info = VersionInfo()
    info.python = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    info.os = platform.platform()

    # PyTorch — via _backend (single import path)
    try:
        torch = get_torch()
        info.pytorch = torch.__version__
    except ImportError:
        pass

    # torch_npu — via _backend (single import path)
    npu = get_torch_npu()
    if npu is not None:
        info.torch_npu = getattr(npu, "__version__", "unknown")

    # CANN version (from environment or torch_npu)
    import os
    cann_home = os.environ.get("ASCEND_HOME_PATH", "")
    if cann_home:
        version_file = os.path.join(cann_home, "version.info")
        try:
            with open(version_file) as f:
                info.cann = f.read().strip()
        except (OSError, IOError):
            pass

    if not info.cann and npu is not None:
        try:
            if hasattr(npu, "version") and hasattr(npu.version, "cann"):
                info.cann = npu.version.cann
        except AttributeError:
            pass

    return info


def _check_python(info: VersionInfo) -> CheckResult:
    major, minor = sys.version_info.major, sys.version_info.minor
    if major != 3 or minor < 8:
        return CheckResult("Python", "error",
                           f"Python {info.python} — requires 3.8+")
    if minor > 12:
        return CheckResult("Python", "warning",
                           f"Python {info.python} — may not be tested with torch_npu")
    return CheckResult("Python", "ok", f"Python {info.python}")


def _check_pytorch(info: VersionInfo) -> CheckResult:
    if not info.pytorch:
        return CheckResult("PyTorch", "error", "PyTorch is not installed")
    major_minor = ".".join(info.pytorch.split(".")[:2])
    if float(major_minor) < 2.0:
        return CheckResult("PyTorch", "error",
                           f"PyTorch {info.pytorch} — requires 2.0+")
    return CheckResult("PyTorch", "ok", f"PyTorch {info.pytorch}")


def _check_torch_npu(info: VersionInfo) -> CheckResult:
    if not info.torch_npu:
        return CheckResult("torch_npu", "warning",
                           "torch_npu not installed — NPU support unavailable",
                           detail="Install: pip install torch-npu (from Huawei's repo)")
    return CheckResult("torch_npu", "ok", f"torch_npu {info.torch_npu}")


def _check_cann(info: VersionInfo) -> CheckResult:
    if not info.cann:
        if info.torch_npu:
            return CheckResult("CANN", "warning",
                               "CANN version not detected — set ASCEND_HOME_PATH")
        return CheckResult("CANN", "skipped", "CANN check skipped (no torch_npu)")
    return CheckResult("CANN", "ok", f"CANN {info.cann}")


def _check_cross_compat(info: VersionInfo) -> CheckResult:
    """Check that torch_npu and PyTorch versions are compatible."""
    if not info.torch_npu or not info.pytorch:
        return CheckResult("Compatibility", "skipped",
                           "Cannot check — missing version info")

    npu_mm = ".".join(info.torch_npu.split(".")[:2])
    pt_mm = ".".join(info.pytorch.split(".")[:2])

    for npu_prefix, pt_prefix, cann_prefix in _COMPAT_MATRIX:
        if npu_mm.startswith(npu_prefix) and pt_mm.startswith(pt_prefix):
            msg = f"torch_npu {info.torch_npu} + PyTorch {info.pytorch} — compatible"
            if info.cann:
                cann_mm = ".".join(info.cann.split(".")[:2])
                if not cann_mm.startswith(cann_prefix):
                    return CheckResult("Compatibility", "warning",
                                       f"CANN {info.cann} may not match torch_npu {info.torch_npu} "
                                       f"(expected CANN {cann_prefix}.*)")
            return CheckResult("Compatibility", "ok", msg)

    # No match found in matrix
    if npu_mm != pt_mm:
        return CheckResult("Compatibility", "error",
                           f"torch_npu {info.torch_npu} and PyTorch {info.pytorch} "
                           f"have different major.minor versions — likely incompatible")

    return CheckResult("Compatibility", "warning",
                       f"Version combination not in known-good matrix — may work but untested")


def _check_npu_device(info: VersionInfo) -> CheckResult:
    """Check if an NPU device is actually usable."""
    if not info.torch_npu:
        return CheckResult("NPU Device", "skipped", "No torch_npu")

    try:
        import torch
        if hasattr(torch, "npu") and torch.npu.is_available():
            count = torch.npu.device_count()
            name = torch.npu.get_device_name(0) if count > 0 else "unknown"
            return CheckResult("NPU Device", "ok",
                               f"{count} NPU(s) available: {name}")
        else:
            return CheckResult("NPU Device", "error",
                               "torch_npu installed but no NPU devices detected",
                               detail="Check: driver installed, npu-smi works, "
                                      "ASCEND_RT_VISIBLE_DEVICES not empty")
    except Exception as exc:  # noqa: BLE001
        return CheckResult("NPU Device", "error",
                           f"NPU device check failed: {exc}")


def format_report(results: List[CheckResult]) -> str:
    """Format check results as a human-readable report."""
    icons = {"ok": "[OK]", "warning": "[!!]", "error": "[XX]", "skipped": "[--]"}
    lines = ["ascend-compat doctor — environment check", "=" * 50]
    for r in results:
        icon = icons.get(r.status, "[??]")
        lines.append(f"  {icon} {r.name}: {r.message}")
        if r.detail:
            lines.append(f"       {r.detail}")
    lines.append("=" * 50)

    errors = sum(1 for r in results if r.status == "error")
    warnings = sum(1 for r in results if r.status == "warning")
    if errors:
        lines.append(f"  {errors} error(s) found — fix these before proceeding")
    elif warnings:
        lines.append(f"  {warnings} warning(s) — may cause issues")
    else:
        lines.append("  All checks passed!")

    return "\n".join(lines)
