"""Layer 4: Diagnostics, validation, and error translation.

Addresses the #1 developer complaint: environment setup.

Provides:
- ``version_check``: Verify CANN ↔ torch_npu ↔ PyTorch ↔ driver compatibility
- ``error_codes``: Translate cryptic CANN error codes to human-readable messages
- ``fallback_monitor``: Detect and report CPU-fallback ops with performance estimates
- ``op_auditor``: Profile a model to predict NPU coverage before deployment
- ``env_setup``: Deep environment validation (CANN dirs, driver, firmware, libs)
- ``security_check``: Verify torch_npu / CANN binary integrity

Usage::

    from ascend_compat.doctor import run_all_checks
    run_all_checks()

    # Or via CLI:
    # ascend-compat doctor
"""

from ascend_compat.doctor.version_check import check_versions
from ascend_compat.doctor.error_codes import translate_error
from ascend_compat.doctor.fallback_monitor import FallbackMonitor
from ascend_compat.doctor.op_auditor import audit_model
from ascend_compat.doctor.env_setup import full_environment_check
from ascend_compat.doctor.security_check import (
    full_security_check,
    verify_torch_npu_integrity,
)

__all__ = [
    "check_versions",
    "translate_error",
    "FallbackMonitor",
    "audit_model",
    "full_environment_check",
    "full_security_check",
    "verify_torch_npu_integrity",
]
