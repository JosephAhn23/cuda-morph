"""Empirical operator validation.

This module provides tools to verify that shimmed operators produce
numerically correct results on real Ascend hardware by comparing against
CPU reference implementations.

**This is the proof layer.** Without these checks passing on actual NPU
hardware, all argument mappings in ``ascend_compat.ecosystem.flash_attn``
are based on documentation, not empirical evidence.

Usage::

    from ascend_compat.validation import OperatorVerifier
    verifier = OperatorVerifier(device="npu")
    results = verifier.run_all()
    print(verifier.format_report(results))
"""

from ascend_compat.validation.op_verifier import OperatorVerifier, VerificationResult

__all__ = ["OperatorVerifier", "VerificationResult"]
