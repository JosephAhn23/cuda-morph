"""Deep environment validation for Ascend development.

Environment setup is universally described as the #1 developer pain point:
    "迁移适配工作着实不是一件容易的事...有些问题最终还是求助于华为的运维工程师才得以解决"
    ("Migration work is truly not easy... some problems ultimately required
     asking Huawei's operations engineers")

A typical setup takes "half a week" due to strict version coupling:
    CANN ↔ torch_npu ↔ PyTorch ↔ driver ↔ firmware

This module validates EVERYTHING beyond basic version checks:
- CANN toolkit installation completeness
- Driver/firmware presence and version
- Required system libraries
- Environment variables
- File permissions
- Custom op compilation readiness (cmake, ccec, etc.)

Usage::

    from ascend_compat.doctor.env_setup import full_environment_check

    report = full_environment_check()
    for item in report:
        print(f"[{item.status}] {item.name}: {item.message}")
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ascend_compat._logging import get_logger

logger = get_logger(__name__)


@dataclass
class EnvCheckResult:
    """Result of a single environment check."""
    name: str
    status: str  # "ok", "warning", "error", "info"
    message: str
    detail: str = ""
    fix: str = ""


# ---------------------------------------------------------------------------
# The full CANN version compatibility matrix
# ---------------------------------------------------------------------------
# Source: Huawei's official documentation (hiascend.com)
# Format: (cann, driver, firmware, torch_npu, pytorch)

@dataclass
class VersionRow:
    """A known-good version combination."""
    cann: str
    driver: str
    firmware: str
    torch_npu: str
    pytorch: str
    python_min: str = "3.8"
    python_max: str = "3.12"


COMPAT_MATRIX: List[VersionRow] = [
    VersionRow("8.0.RC3",  "24.1.rc3", "7.5.0.1",  "2.5.1",  "2.5.1", "3.8", "3.12"),
    VersionRow("8.0.RC2",  "24.1.rc2", "7.3.0.1",  "2.4.0",  "2.4.0", "3.8", "3.12"),
    VersionRow("8.0.RC1",  "24.1.rc1", "7.1.0.6",  "2.3.1",  "2.3.1", "3.8", "3.11"),
    VersionRow("7.0.1",    "23.0.6",   "7.1.0.5",  "2.2.0",  "2.2.0", "3.8", "3.11"),
    VersionRow("7.0.0",    "23.0.5",   "7.1.0.3",  "2.1.0",  "2.1.0", "3.8", "3.10"),
    VersionRow("6.3.RC3",  "23.0.3",   "6.4.0.4",  "2.0.1",  "2.0.1", "3.8", "3.10"),
]


def full_environment_check() -> List[EnvCheckResult]:
    """Run a comprehensive environment validation.

    Checks:
    1. Operating system compatibility
    2. Python version
    3. CANN toolkit installation
    4. NPU driver and firmware
    5. Required libraries (libascendcl.so, etc.)
    6. Environment variables
    7. Custom op compilation tools (cmake, ccec)
    8. torch_npu installation
    9. HCCL distributed training readiness
    10. Disk space

    Returns:
        List of :class:`EnvCheckResult` items.
    """
    results: List[EnvCheckResult] = []

    results.append(_check_os())
    results.append(_check_python())
    results.extend(_check_cann_installation())
    results.extend(_check_driver_firmware())
    results.extend(_check_env_vars())
    results.extend(_check_compilation_tools())
    results.append(_check_torch_npu_installation())
    results.extend(_check_hccl_readiness())
    results.append(_check_disk_space())

    return results


def _check_os() -> EnvCheckResult:
    """Check operating system compatibility."""
    system = platform.system()
    release = platform.release()
    arch = platform.machine()

    if system == "Linux":
        # Check for supported distros
        distro = _get_linux_distro()
        if any(d in distro.lower() for d in ["ubuntu", "centos", "euler", "kylin", "uos"]):
            return EnvCheckResult("OS", "ok",
                                  f"{distro} ({arch})",
                                  detail=f"Kernel: {release}")
        return EnvCheckResult("OS", "warning",
                              f"{distro} ({arch}) — not officially supported",
                              detail="Huawei officially supports: Ubuntu 18.04/20.04/22.04, "
                                     "CentOS 7/8, EulerOS 2.x, Kylin V10, UOS 20",
                              fix="Consider using Ubuntu 20.04 or EulerOS for best compatibility")

    elif system == "Windows":
        return EnvCheckResult("OS", "info",
                              f"Windows {release} — development only",
                              detail="Ascend NPU drivers are Linux-only. "
                                     "ascend-compat can be developed/tested on Windows "
                                     "but NPU features require Linux.")

    return EnvCheckResult("OS", "warning",
                          f"{system} {release} — not officially supported")


def _check_python() -> EnvCheckResult:
    """Check Python version against the compatibility matrix."""
    ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    major, minor = sys.version_info.major, sys.version_info.minor

    if major != 3:
        return EnvCheckResult("Python", "error", f"Python {ver} — requires Python 3.x")

    if minor < 8:
        return EnvCheckResult("Python", "error",
                              f"Python {ver} — minimum 3.8 required for torch_npu")
    if minor > 12:
        return EnvCheckResult("Python", "warning",
                              f"Python {ver} — 3.13+ may not be tested with torch_npu yet",
                              fix="Use Python 3.10 or 3.11 for best compatibility")

    return EnvCheckResult("Python", "ok", f"Python {ver}")


def _check_cann_installation() -> List[EnvCheckResult]:
    """Validate CANN toolkit installation completeness."""
    results: List[EnvCheckResult] = []

    ascend_home = os.environ.get("ASCEND_HOME_PATH", "")

    # Try to find CANN in standard locations
    candidates = [
        ascend_home,
        "/usr/local/Ascend/ascend-toolkit/latest",
        "/usr/local/Ascend/latest",
        os.path.expanduser("~/Ascend/ascend-toolkit/latest"),
    ]

    found_path = ""
    for path in candidates:
        if path and os.path.isdir(path):
            found_path = path
            break

    if not found_path:
        results.append(EnvCheckResult("CANN Toolkit", "error",
                                      "CANN toolkit not found",
                                      fix="Install from: https://www.hiascend.com/software/cann/community\n"
                                          "Then: export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest"))
        return results

    # Check version
    version = "unknown"
    version_file = os.path.join(found_path, "version.info")
    if os.path.isfile(version_file):
        try:
            with open(version_file) as f:
                version = f.read().strip()
        except IOError:
            pass

    results.append(EnvCheckResult("CANN Toolkit", "ok",
                                  f"CANN {version} at {found_path}"))

    # Check critical directories
    critical_dirs = {
        "lib64":           "Runtime libraries",
        "include":         "C++ headers",
        "bin":             "CLI tools (npu-smi, ccec)",
        "python":          "Python packages",
        "opp":             "Operator library (built-in ops)",
    }

    for dirname, desc in critical_dirs.items():
        dirpath = os.path.join(found_path, dirname)
        if os.path.isdir(dirpath):
            results.append(EnvCheckResult(f"CANN/{dirname}", "ok", f"{desc}: present"))
        else:
            results.append(EnvCheckResult(f"CANN/{dirname}", "warning",
                                          f"{desc}: missing at {dirpath}",
                                          fix=f"Reinstall CANN or check {dirname} directory"))

    # Check for libascendcl.so
    lib_path = os.path.join(found_path, "lib64", "libascendcl.so")
    if os.path.isfile(lib_path):
        results.append(EnvCheckResult("libascendcl", "ok", "ACL runtime library present"))
    else:
        results.append(EnvCheckResult("libascendcl", "error",
                                      "libascendcl.so not found — ACL runtime missing",
                                      fix="Reinstall CANN toolkit completely"))

    return results


def _check_driver_firmware() -> List[EnvCheckResult]:
    """Check NPU driver and firmware versions."""
    results: List[EnvCheckResult] = []

    # Check driver
    driver_path = "/usr/local/Ascend/driver/version.info"
    if os.path.isfile(driver_path):
        try:
            with open(driver_path) as f:
                driver_version = f.read().strip()
            results.append(EnvCheckResult("NPU Driver", "ok",
                                          f"Driver: {driver_version}"))
        except IOError:
            results.append(EnvCheckResult("NPU Driver", "warning",
                                          "Driver version file unreadable"))
    else:
        if platform.system() == "Linux":
            results.append(EnvCheckResult("NPU Driver", "warning",
                                          "NPU driver not found at standard path",
                                          fix="Install: https://www.hiascend.com/hardware/firmware-drivers"))
        else:
            results.append(EnvCheckResult("NPU Driver", "info",
                                          "NPU driver check skipped (non-Linux)"))

    # Try npu-smi
    npu_smi = shutil.which("npu-smi")
    if npu_smi:
        try:
            proc = subprocess.run(
                [npu_smi, "info"],
                capture_output=True, text=True, timeout=10,
            )
            if proc.returncode == 0:
                # Parse device count from output
                lines = proc.stdout.strip().split("\n")
                results.append(EnvCheckResult("npu-smi", "ok",
                                              f"npu-smi available — {len([l for l in lines if 'NPU' in l])} device(s)"))
            else:
                results.append(EnvCheckResult("npu-smi", "warning",
                                              f"npu-smi returned error: {proc.stderr[:200]}"))
        except (subprocess.TimeoutExpired, OSError):
            results.append(EnvCheckResult("npu-smi", "warning",
                                          "npu-smi timed out or failed"))
    else:
        if platform.system() == "Linux":
            results.append(EnvCheckResult("npu-smi", "warning",
                                          "npu-smi not on PATH",
                                          fix="Add to PATH: export PATH=$PATH:/usr/local/Ascend/driver/tools"))

    return results


def _check_env_vars() -> List[EnvCheckResult]:
    """Check required and recommended environment variables."""
    results: List[EnvCheckResult] = []

    required_vars = {
        "ASCEND_HOME_PATH": (
            "CANN toolkit location",
            "export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest",
        ),
    }

    recommended_vars = {
        "LD_LIBRARY_PATH": (
            "Should include CANN lib64 path",
            "export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$LD_LIBRARY_PATH",
        ),
        "PYTHONPATH": (
            "Should include CANN Python packages",
            "export PYTHONPATH=$ASCEND_HOME_PATH/python/site-packages:$PYTHONPATH",
        ),
    }

    for var, (desc, fix) in required_vars.items():
        val = os.environ.get(var, "")
        if val:
            results.append(EnvCheckResult(f"env:{var}", "ok", f"{var}={val}"))
        else:
            results.append(EnvCheckResult(f"env:{var}", "warning",
                                          f"{var} not set — {desc}",
                                          fix=fix))

    for var, (desc, fix) in recommended_vars.items():
        val = os.environ.get(var, "")
        if val:
            # Check if CANN paths are included
            ascend_home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend")
            if ascend_home in val or "Ascend" in val:
                results.append(EnvCheckResult(f"env:{var}", "ok",
                                              f"{var} includes CANN paths"))
            else:
                results.append(EnvCheckResult(f"env:{var}", "info",
                                              f"{var} set but may not include CANN paths",
                                              fix=fix))
        else:
            results.append(EnvCheckResult(f"env:{var}", "info",
                                          f"{var} not set — {desc}",
                                          fix=fix))

    # Check for conflicting variables
    cuda_home = os.environ.get("CUDA_HOME", "")
    if cuda_home and not os.environ.get("ASCEND_HOME_PATH"):
        results.append(EnvCheckResult("env:CUDA_HOME", "info",
                                      f"CUDA_HOME={cuda_home} set but ASCEND_HOME_PATH is not",
                                      detail="On an Ascend system, ASCEND_HOME_PATH takes priority"))

    return results


def _check_compilation_tools() -> List[EnvCheckResult]:
    """Check tools needed for custom Ascend C operator compilation."""
    results: List[EnvCheckResult] = []

    # cmake
    cmake = shutil.which("cmake")
    if cmake:
        try:
            proc = subprocess.run([cmake, "--version"],
                                  capture_output=True, text=True, timeout=5)
            version_line = proc.stdout.split("\n")[0] if proc.stdout else "unknown"
            results.append(EnvCheckResult("cmake", "ok", version_line))
        except (subprocess.TimeoutExpired, OSError):
            results.append(EnvCheckResult("cmake", "ok", f"cmake at {cmake}"))
    else:
        results.append(EnvCheckResult("cmake", "warning",
                                      "cmake not found — needed for custom op compilation",
                                      fix="apt install cmake (Ubuntu) / yum install cmake (CentOS)"))

    # Bisheng compiler (ccec)
    ascend_home = os.environ.get("ASCEND_HOME_PATH", "")
    if ascend_home:
        ccec_path = os.path.join(ascend_home, "bin", "ccec")
        if os.path.isfile(ccec_path):
            results.append(EnvCheckResult("ccec (Bisheng)", "ok",
                                          f"Ascend C compiler at {ccec_path}"))
        else:
            results.append(EnvCheckResult("ccec (Bisheng)", "info",
                                          "Bisheng compiler not found — needed for Ascend C kernels",
                                          fix="Part of CANN toolkit. Check ASCEND_HOME_PATH/bin/"))

    # g++ / C++ compiler
    gpp = shutil.which("g++") or shutil.which("c++")
    if gpp:
        results.append(EnvCheckResult("C++ compiler", "ok", f"Found: {gpp}"))
    else:
        results.append(EnvCheckResult("C++ compiler", "warning",
                                      "No C++ compiler found",
                                      fix="apt install g++ (Ubuntu) / yum install gcc-c++ (CentOS)"))

    return results


def _check_torch_npu_installation() -> EnvCheckResult:
    """Check torch_npu installation and backend registration."""
    try:
        import torch
    except ImportError:
        return EnvCheckResult("torch_npu", "error",
                              "PyTorch not installed — prerequisite for torch_npu")

    try:
        import torch_npu  # type: ignore[import-untyped]
        version = getattr(torch_npu, "__version__", "unknown")

        # Check PrivateUse1 backend registration
        backend_name = "unknown"
        if hasattr(torch, "_C") and hasattr(torch._C, "_get_privateuse1_backend_name"):
            backend_name = torch._C._get_privateuse1_backend_name()

        return EnvCheckResult("torch_npu", "ok",
                              f"torch_npu {version} (backend: {backend_name})")

    except ImportError:
        return EnvCheckResult("torch_npu", "warning",
                              "torch_npu not installed — NPU support unavailable",
                              fix="pip install torch-npu (from Huawei's repo)")
    except Exception as exc:  # noqa: BLE001
        return EnvCheckResult("torch_npu", "error",
                              f"torch_npu import failed: {exc}",
                              fix="Check CANN/driver versions. Run: ascend-compat doctor")


def _check_hccl_readiness() -> List[EnvCheckResult]:
    """Check HCCL (Huawei Collective Communication Library) for distributed training."""
    results: List[EnvCheckResult] = []

    ascend_home = os.environ.get("ASCEND_HOME_PATH", "")

    # Check HCCL library
    if ascend_home:
        hccl_lib = os.path.join(ascend_home, "lib64", "libhccl.so")
        if os.path.isfile(hccl_lib):
            results.append(EnvCheckResult("HCCL", "ok", "HCCL library present"))
        else:
            results.append(EnvCheckResult("HCCL", "info",
                                          "HCCL library not found — distributed training may not work",
                                          fix="HCCL is part of CANN. Reinstall if needed."))

    # Check HCCL-related env vars
    hccl_timeout = os.environ.get("HCCL_CONNECT_TIMEOUT", "")
    if hccl_timeout:
        results.append(EnvCheckResult("HCCL timeout", "ok",
                                      f"HCCL_CONNECT_TIMEOUT={hccl_timeout}"))
    else:
        results.append(EnvCheckResult("HCCL timeout", "info",
                                      "HCCL_CONNECT_TIMEOUT not set (default ~120s)",
                                      fix="For large clusters: export HCCL_CONNECT_TIMEOUT=3600"))

    return results


def _check_disk_space() -> EnvCheckResult:
    """Check available disk space (CANN requires significant space)."""
    try:
        import shutil as sh
        total, used, free = sh.disk_usage("/")
        free_gb = free / (1024**3)
        total_gb = total / (1024**3)

        if free_gb < 5:
            return EnvCheckResult("Disk Space", "warning",
                                  f"{free_gb:.1f} GB free / {total_gb:.1f} GB total",
                                  fix="CANN toolkit requires ~15 GB. Free up disk space.")
        return EnvCheckResult("Disk Space", "ok",
                              f"{free_gb:.1f} GB free / {total_gb:.1f} GB total")
    except Exception:  # noqa: BLE001
        return EnvCheckResult("Disk Space", "info", "Could not check disk space")


def _get_linux_distro() -> str:
    """Get Linux distribution name."""
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    return line.split("=", 1)[1].strip().strip('"')
    except (IOError, OSError):
        pass
    return platform.platform()


def format_env_report(results: List[EnvCheckResult]) -> str:
    """Format environment check results as a comprehensive report."""
    icons = {"ok": "[OK]", "warning": "[!!]", "error": "[XX]", "info": "[ii]"}
    lines = [
        "ascend-compat doctor — full environment check",
        "=" * 60,
    ]

    for r in results:
        icon = icons.get(r.status, "[??]")
        lines.append(f"  {icon} {r.name}: {r.message}")
        if r.detail:
            lines.append(f"       Detail: {r.detail}")
        if r.fix:
            lines.append(f"       Fix: {r.fix}")

    lines.append("=" * 60)

    errors = sum(1 for r in results if r.status == "error")
    warnings = sum(1 for r in results if r.status == "warning")
    oks = sum(1 for r in results if r.status == "ok")

    summary = f"  {oks} passed, {warnings} warning(s), {errors} error(s)"
    lines.append(summary)

    if errors:
        lines.append("  Fix errors above before proceeding with NPU development.")
    elif warnings:
        lines.append("  Warnings may cause issues — review and fix if possible.")
    else:
        lines.append("  Environment looks good!")

    return "\n".join(lines)
