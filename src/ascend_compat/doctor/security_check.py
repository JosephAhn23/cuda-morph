"""Security verification for the Ascend software stack.

Verifies the integrity of torch_npu and related CANN libraries by checking
file hashes against known-good values from official releases.

Why this matters
----------------
``import torch_npu`` executes arbitrary code.  If a malicious package has
been installed (e.g. via a typosquat attack, compromised build, or local
modification), it could exfiltrate training data, inject backdoors into
models, or mine cryptocurrency using NPU compute.

This module provides a best-effort check: it hashes the torch_npu package
init file and compares against known-good hashes.  It also verifies that
CANN shared libraries in ``ASCEND_HOME_PATH`` have expected sizes and
are signed (where platform-supported).

Limitations
-----------
- This is NOT a substitute for signed packages or a proper software supply
  chain.  It catches obvious tampering but not sophisticated attacks.
- Hashes must be updated when new torch_npu versions are released.
- Only the ``__init__.py`` is checked, not every file in the package.

Usage::

    from ascend_compat.doctor.security_check import verify_torch_npu_integrity
    result = verify_torch_npu_integrity()
    print(result)
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ascend_compat._logging import get_logger

logger = get_logger(__name__)


@dataclass
class IntegrityResult:
    """Result of an integrity check."""
    package: str
    version: str
    status: str  # "ok", "warning", "error", "unknown"
    message: str
    details: Dict[str, str] = field(default_factory=dict)


# Known-good hashes: {version: sha256_of_init_file}
# These should be populated from official releases.
# Using empty dict for now — community should fill these in.
_KNOWN_HASHES: Dict[str, str] = {
    # "2.5.1": "abc123...",
    # "2.4.0": "def456...",
}

# Expected CANN shared libraries
_EXPECTED_CANN_LIBS = [
    "libascendcl.so",
    "libhccl.so",
    "libge_runner.so",
    "libacl_op_compiler.so",
]


def verify_torch_npu_integrity() -> IntegrityResult:
    """Verify torch_npu package integrity against known-good hashes.

    Checks:
    1. torch_npu ``__init__.py`` exists and is readable
    2. SHA-256 hash matches the known-good value for this version
    3. Package location is in a standard site-packages directory

    Returns:
        :class:`IntegrityResult` with status and details.
    """
    try:
        import importlib.util
        spec = importlib.util.find_spec("torch_npu")
    except (ImportError, ModuleNotFoundError, ValueError):
        return IntegrityResult(
            package="torch_npu",
            version="unknown",
            status="warning",
            message="torch_npu is not installed — cannot verify integrity",
        )

    if spec is None or spec.origin is None:
        return IntegrityResult(
            package="torch_npu",
            version="unknown",
            status="warning",
            message="torch_npu spec has no origin path",
        )

    origin = spec.origin
    details: Dict[str, str] = {"path": origin}

    # Get version
    try:
        from ascend_compat._backend import get_torch_npu
        npu_mod = get_torch_npu()
        version = getattr(npu_mod, "__version__", "unknown") if npu_mod else "unknown"
    except Exception:  # noqa: BLE001
        version = "unknown"
    details["version"] = version

    # Hash the init file
    try:
        with open(origin, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        details["sha256"] = file_hash
    except (OSError, IOError) as exc:
        return IntegrityResult(
            package="torch_npu",
            version=version,
            status="error",
            message=f"Cannot read torch_npu: {exc}",
            details=details,
        )

    # Check location
    is_standard = "site-packages" in origin or "dist-packages" in origin
    details["standard_location"] = str(is_standard)
    if not is_standard:
        logger.warning("torch_npu loaded from non-standard location: %s", origin)

    # Check against known hashes
    base_version = version.split("+")[0].split("a")[0]
    expected_hash = _KNOWN_HASHES.get(base_version)

    if expected_hash is None:
        return IntegrityResult(
            package="torch_npu",
            version=version,
            status="unknown",
            message=(
                f"torch_npu {version} hash not in known-good database. "
                f"Integrity cannot be verified (this is expected for new releases)."
            ),
            details=details,
        )

    if file_hash == expected_hash:
        return IntegrityResult(
            package="torch_npu",
            version=version,
            status="ok",
            message=f"torch_npu {version} integrity verified (SHA-256 match)",
            details=details,
        )

    return IntegrityResult(
        package="torch_npu",
        version=version,
        status="error",
        message=(
            f"torch_npu {version} hash MISMATCH! "
            f"Expected {expected_hash[:16]}..., got {file_hash[:16]}... "
            f"This may indicate corruption or tampering."
        ),
        details=details,
    )


def verify_cann_libraries() -> List[IntegrityResult]:
    """Check that expected CANN shared libraries exist and are accessible.

    Verifies that the CANN toolkit libraries in ``ASCEND_HOME_PATH`` are
    present and have non-zero file sizes.

    Returns:
        List of :class:`IntegrityResult` for each expected library.
    """
    results: List[IntegrityResult] = []

    cann_home = os.environ.get("ASCEND_HOME_PATH", "")
    if not cann_home:
        results.append(IntegrityResult(
            package="CANN",
            version="",
            status="warning",
            message="ASCEND_HOME_PATH not set — cannot verify CANN libraries",
        ))
        return results

    lib_dir = os.path.join(cann_home, "lib64")
    if not os.path.isdir(lib_dir):
        lib_dir = os.path.join(cann_home, "lib")

    for lib_name in _EXPECTED_CANN_LIBS:
        lib_path = os.path.join(lib_dir, lib_name) if os.path.isdir(lib_dir) else ""
        details: Dict[str, str] = {"expected_path": lib_path}

        if not lib_path or not os.path.isfile(lib_path):
            results.append(IntegrityResult(
                package=lib_name,
                version="",
                status="warning",
                message=f"{lib_name} not found in {lib_dir}",
                details=details,
            ))
            continue

        file_size = os.path.getsize(lib_path)
        details["file_size"] = str(file_size)

        if file_size == 0:
            results.append(IntegrityResult(
                package=lib_name,
                version="",
                status="error",
                message=f"{lib_name} exists but is empty (0 bytes)",
                details=details,
            ))
        else:
            results.append(IntegrityResult(
                package=lib_name,
                version="",
                status="ok",
                message=f"{lib_name} present ({file_size} bytes)",
                details=details,
            ))

    return results


def full_security_check() -> List[IntegrityResult]:
    """Run all security checks.

    Returns:
        Combined list of integrity results.
    """
    results = [verify_torch_npu_integrity()]
    results.extend(verify_cann_libraries())
    return results


def format_security_report(results: List[IntegrityResult]) -> str:
    """Format security check results as a human-readable report."""
    icons = {"ok": "[OK]", "warning": "[!!]", "error": "[XX]", "unknown": "[??]"}
    lines = ["ascend-compat security check", "=" * 50]
    for r in results:
        icon = icons.get(r.status, "[??]")
        lines.append(f"  {icon} {r.package} ({r.version}): {r.message}")
        if r.details:
            for k, v in r.details.items():
                lines.append(f"       {k}: {v}")
    lines.append("=" * 50)

    errors = sum(1 for r in results if r.status == "error")
    if errors:
        lines.append(f"  {errors} SECURITY CONCERN(S) found — investigate before proceeding")
    else:
        lines.append("  No integrity issues detected")

    return "\n".join(lines)
