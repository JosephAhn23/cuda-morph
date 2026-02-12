"""CANN error code translator.

CANN (Compute Architecture for Neural Networks) produces error codes that are
notoriously cryptic.  Developers frequently encounter codes like ``507035``,
``507008``, or ``ERR99999 UNKNOWN`` with no public documentation explaining
what went wrong.

This module maintains a database of the ~50 most common CANN error codes
(compiled from CSDN, Zhihu, GitHub/Gitee issues, and Huawei support forums)
and provides human-readable translations with suggested fixes.

Usage::

    from ascend_compat.doctor.error_codes import translate_error

    msg = translate_error(507035)
    print(msg)
    # → "CANN 507035: Operator execution failed — internal kernel error.
    #    Likely cause: unsupported dtype or tensor shape for this op.
    #    Fix: Check input dtypes (FP64 not supported), ensure tensor
    #    dimensions meet alignment requirements (multiples of 16 for Cube ops)."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ErrorInfo:
    """Human-readable information about a CANN error code."""

    code: str
    category: str       # "runtime", "compile", "memory", "driver", "operator", "environment"
    summary: str         # One-line description
    likely_cause: str    # Most common root cause
    fix: str             # Suggested fix
    references: str = "" # Links to relevant docs/issues
    cann_versions: str = ""  # CANN versions where this error is known (e.g. "7.0+", "8.0.RC1-RC3")
    added_in: str = ""       # ascend-compat version that added this entry


# ---------------------------------------------------------------------------
# Error code database
# ---------------------------------------------------------------------------

_ERROR_DB: Dict[str, ErrorInfo] = {}


def _add(
    code: str, category: str, summary: str, cause: str, fix: str,
    refs: str = "", cann_versions: str = "", added_in: str = "0.2.0",
) -> None:
    _ERROR_DB[str(code)] = ErrorInfo(
        code=str(code), category=category, summary=summary,
        likely_cause=cause, fix=fix, references=refs,
        cann_versions=cann_versions, added_in=added_in,
    )


# ── Runtime errors ──────────────────────────────────────────────────────

_add("507008", "runtime",
     "ACL stream synchronize failed",
     "Device-side error during kernel execution (often OOM or invalid op)",
     "Check memory usage with `ascend-compat doctor`. Reduce batch size. "
     "Verify all ops are NPU-compatible with `ascend-compat check`.",
     cann_versions="6.3+", added_in="0.2.0")

_add("507011", "runtime",
     "Device memory allocation failed",
     "Out of device memory (NPU VRAM exhausted)",
     "Reduce batch size, enable gradient checkpointing, use mixed precision (FP16). "
     "Check memory with torch.npu.memory_summary().")

_add("507015", "runtime",
     "ACL runtime internal error",
     "CANN runtime encountered an unexpected state",
     "Restart the process. If persistent, check CANN/driver version compatibility. "
     "Run `ascend-compat doctor` to verify versions.")

_add("507035", "operator",
     "Operator execution failed — internal kernel error",
     "Unsupported dtype (e.g. FP64) or tensor shape doesn't meet alignment requirements",
     "Check input dtypes — use FP16 or FP32 (no FP64 on Ascend 910A). "
     "Ensure matrix dimensions are multiples of 16 for Cube Unit ops. "
     "Check for NaN/Inf in inputs.",
     cann_versions="6.3+", added_in="0.2.0")

_add("507037", "operator",
     "Operator input validation failed",
     "Input tensor shape, dtype, or format is invalid for this operator",
     "Check the CANN op specification for valid input ranges. "
     "Common issue: empty tensors, zero-dim tensors, or unsupported dtypes.")

# ── Memory errors ──────────────────────────────────────────────────────

_add("207001", "memory",
     "Memory allocation failed on host",
     "System RAM exhausted (not device memory)",
     "Free host memory. Check for memory leaks in data loading pipeline. "
     "Use smaller datasets or memory-mapped files.")

_add("207006", "memory",
     "DMA (Direct Memory Access) transfer failed",
     "Host-to-device or device-to-host data transfer error",
     "Check pin_memory settings. Ensure HBMM driver is functioning. "
     "Restart the NPU device if needed: `npu-smi reset`.")

# ── Compile/graph errors ────────────────────────────────────────────────

_add("500001", "compile",
     "Graph compilation failed",
     "CANN cannot compile the computation graph (unsupported op or shape)",
     "Check for dynamic shapes (use torch.compile with dynamic=False). "
     "Some ops may need to be excluded from graph capture.")

_add("500002", "compile",
     "Operator not found in CANN op library",
     "The operation is not implemented in the current CANN version",
     "Check CANN version compatibility. Update CANN to the latest version. "
     "As a workaround, the op will fall back to CPU (slow).")

# ── Driver errors ──────────────────────────────────────────────────────

_add("107001", "driver",
     "NPU device not found",
     "No Ascend NPU detected by the driver",
     "Run `npu-smi info` to check device status. "
     "Verify driver installation: `cat /usr/local/Ascend/driver/version.info`. "
     "Check ASCEND_RT_VISIBLE_DEVICES is not set to empty.")

_add("107002", "driver",
     "NPU device already in use (exclusive mode)",
     "Another process has exclusive access to the NPU",
     "Check running processes: `npu-smi info -t usages`. "
     "Kill orphan processes or use a different device index.")

_add("107005", "driver",
     "NPU driver version mismatch",
     "Installed CANN expects a different driver version",
     "Update driver to match CANN version. Check compatibility matrix "
     "at https://www.hiascend.com/document")

# ── The dreaded ERR99999 ────────────────────────────────────────────────

_add("99999", "runtime",
     "ERR99999: UNKNOWN application exception",
     "Catch-all error — the actual cause is masked. Most common: "
     "(1) version mismatch between CANN/driver/firmware, "
     "(2) accessing uninitialized device memory, "
     "(3) race condition in multi-stream execution",
     "Run `ascend-compat doctor` to check versions. "
     "Enable CANN debug logging: export ASCEND_GLOBAL_LOG_LEVEL=1. "
     "Check dmesg for kernel-level NPU errors. "
     "If reproducible, file a bug at https://gitee.com/ascend/pytorch/issues",
     cann_versions="6.3+", added_in="0.2.0")

_add("ERR99999", "runtime",
     "ERR99999: UNKNOWN application exception",
     "Same as error code 99999 — this is the string form",
     "See error code 99999 for diagnosis steps.")

# ── Common PyTorch-level errors (not CANN codes but frequent) ──────────

_add("CUDA_NOT_COMPILED", "runtime",
     "AssertionError: Torch not compiled with CUDA enabled",
     "Code calls torch.cuda directly on an Ascend system without the shim",
     "Add `import ascend_compat` at the top of your script. "
     "Or: replace `torch.cuda` calls with `torch.npu` equivalents.")

_add("NCCL_UNAVAILABLE", "distributed",
     "NCCL backend is not available",
     "DeepSpeed or accelerate tried to use NCCL which doesn't exist on Ascend",
     "Use HCCL backend: `dist.init_process_group(backend='hccl')`. "
     "Or: `from ascend_compat.ecosystem import deepspeed_patch; deepspeed_patch.apply()`")

# ── Additional runtime errors from community reports ──────────────────

_add("507001", "runtime",
     "ACL device set failed",
     "Invalid device index or device not initialized",
     "Check ASCEND_RT_VISIBLE_DEVICES. Verify device index < device_count. "
     "Run `npu-smi info` to list available devices.")

_add("507002", "runtime",
     "ACL context creation failed",
     "Device is in an error state or exclusive-mode conflict",
     "Reset the device: `npu-smi reset -d <id>`. Check for orphan processes.")

_add("507005", "runtime",
     "ACL event synchronize failed",
     "Event signaled an error during asynchronous execution",
     "Often follows a kernel error. Check earlier logs for operator failures. "
     "Enable debug: export ASCEND_GLOBAL_LOG_LEVEL=1")

_add("507014", "runtime",
     "ACL profiling error",
     "Profiler failed to start or record data",
     "Ensure msprof is properly configured. Check file permissions on "
     "profiling output directory.")

_add("507018", "runtime",
     "Resource not released — potential memory leak",
     "Streams, events, or device memory were not freed",
     "Ensure all `torch.npu.Stream()` and events are properly closed. "
     "Use context managers. In long training runs, call `torch.npu.empty_cache()` periodically.")

_add("507021", "runtime",
     "HCCL communication error",
     "Distributed training collective operation failed",
     "Check network connectivity between nodes. Verify HCCL_CONNECT_TIMEOUT is sufficient. "
     "Ensure all ranks are using the same HCCL version.")

_add("507023", "runtime",
     "HCCL timeout — rank did not respond",
     "One or more distributed training ranks timed out during a collective operation",
     "Increase HCCL timeout: `export HCCL_CONNECT_TIMEOUT=3600`. "
     "Check for load imbalance across ranks. Verify all nodes launched simultaneously.")

# ── Operator errors (expanded from vllm-ascend experience) ────────────

_add("507030", "operator",
     "Operator not registered in CANN op library",
     "The requested operation has no CANN implementation",
     "Check CANN version — newer versions add more ops. "
     "The op will fall back to CPU (slow). Consider writing an Ascend C custom op: "
     "`from ascend_compat.kernel_helper import scaffold`")

_add("507032", "operator",
     "Operator shape inference failed",
     "CANN cannot infer output shapes from inputs",
     "Check for dynamic shapes. Try torch.compile with dynamic=False. "
     "Ensure batch dimensions are consistent.")

_add("507034", "operator",
     "Operator dtype not supported",
     "The operation does not support the given data type",
     "Ascend 910A: no FP64, limited BF16. Ascend 910B+: no FP8. "
     "Use ascend_compat's dtype_manager for auto-substitution: "
     "`from ascend_compat.cuda_shim.dtype_manager import enable_auto_dtype`")

_add("507036", "operator",
     "Operator format mismatch (NZ/ND format conflict)",
     "Tensor internal format doesn't match operator expectation. "
     "Ascend uses 5D NC1HWC0 (NZ) format internally; some ops require ND (contiguous)",
     "Call `.contiguous()` on inputs. For custom ops, specify FORMAT_ND in op registration. "
     "Use `npu_format_cast` to explicitly convert tensor format.")

_add("507038", "operator",
     "Tiling computation failed",
     "Host-side tiling for the operator could not fit data into Local Memory",
     "Reduce tensor size per dimension. Ensure alignment (32-byte boundary). "
     "For custom operators, check tiling logic handles edge cases.")

_add("507039", "operator",
     "AICPU operator timeout",
     "Operator running on AICPU (fallback CPU on the NPU chip) timed out",
     "AICPU ops are much slower than AICore ops. Consider using AICore alternatives. "
     "Increase timeout: export AICPU_TIMEOUT=600")

# ── Memory errors (expanded) ─────────────────────────────────────────

_add("207002", "memory",
     "Device memory fragmentation — allocation failed despite free memory",
     "Total free memory is sufficient but no contiguous block is large enough",
     "Call `torch.npu.empty_cache()` to release cached blocks. "
     "Reduce peak memory usage. Consider gradient checkpointing.")

_add("207003", "memory",
     "Host-device memory mapping failed",
     "Unified/pinned memory allocation error",
     "Reduce pin_memory usage in DataLoader. Check ulimit -l (locked memory limit).")

_add("207007", "memory",
     "HBM ECC error detected",
     "Hardware memory error (ECC correction/detection)",
     "This is a hardware issue. Reset the device: `npu-smi reset -d <id>`. "
     "If persistent, contact hardware support — the HBM module may be failing.")

# ── Compile/graph errors (expanded) ──────────────────────────────────

_add("500003", "compile",
     "Graph optimization pass failed",
     "A CANN graph optimization could not be applied",
     "Try disabling specific optimizations. Check for unsupported op combinations. "
     "This often occurs with models that mix static and dynamic shapes.")

_add("500004", "compile",
     "ONNX opset version too high",
     "CANN runtime supports up to ONNX opset 15; model requires opset 18+",
     "Workaround: export model with opset_version=15. "
     "For operators like LayerNormalization, use decomposed equivalents.",
     cann_versions="7.0–8.0", added_in="0.3.0")

_add("500005", "compile",
     "torchair compilation error — Triton kernel not supported",
     "torch.compile via torchair failed on a Triton kernel",
     "Replace Triton kernels with CANN-native ops. Or use Triton-Ascend backend: "
     "`from ascend_compat.ecosystem.triton_bridge import configure_triton_backend`")

# ── FP8/Quantization errors ──────────────────────────────────────────

_add("FP8_UNSUPPORTED", "operator",
     "FP8 dtype not supported on current hardware",
     "Ascend 910A/B/C has no FP8 hardware. FP8 requires Ascend 950 (expected 2026)",
     "Workaround: Re-quantize model to W8A8 or run in BF16/FP16. "
     "Use: `from ascend_compat.cuda_shim.quantization import check_quant_method`",
     cann_versions="7.0+ (all 910 variants)", added_in="0.3.0")

_add("QUINT8_UNSUPPORTED", "operator",
     "torch.quint8 is not supported on Ascend",
     "PyTorch's quantized uint8 type has no Ascend mapping",
     "Use INT8 (W8A8) quantization instead of QUINT8. "
     "Dynamic quantization with torch.quantization should use int8.")

# ── Environment setup errors ─────────────────────────────────────────

_add("CANN_NOT_FOUND", "environment",
     "CANN toolkit not found",
     "ASCEND_HOME_PATH not set and CANN not in standard locations",
     "Install CANN toolkit: https://www.hiascend.com/software/cann/community. "
     "Set: export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest")

_add("DRIVER_MISMATCH", "environment",
     "NPU driver version incompatible with CANN",
     "The installed NPU driver does not match the CANN toolkit version",
     "Update driver to match CANN. Check compatibility: "
     "https://www.hiascend.com/document → 'Version Dependency'")

_add("FIRMWARE_MISMATCH", "environment",
     "NPU firmware version incompatible",
     "Firmware on the NPU chip doesn't match the installed driver/CANN",
     "Update firmware: `npu-smi update -t firmware`. "
     "Requires matching driver version. Reboot after update.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def translate_error(code: any) -> Optional[ErrorInfo]:
    """Translate a CANN error code to human-readable information.

    Args:
        code: Error code (int or string).  Handles both numeric codes
            like 507035 and string codes like "ERR99999".

    Returns:
        :class:`ErrorInfo` with diagnosis and fix, or None if unknown.
    """
    return _ERROR_DB.get(str(code))


def format_error(code: any) -> str:
    """Format an error code as a human-readable string.

    Args:
        code: Error code (int or string).

    Returns:
        Formatted error description, or "Unknown error code" message.
    """
    info = translate_error(code)
    if info is None:
        return (
            f"Unknown CANN error code: {code}\n"
            "  Check CANN docs: https://www.hiascend.com/document\n"
            "  Or search: https://gitee.com/ascend/pytorch/issues"
        )

    lines = [
        f"CANN {info.code}: {info.summary}",
        f"  Category:     {info.category}",
        f"  Likely cause:  {info.likely_cause}",
        f"  Fix:           {info.fix}",
    ]
    if info.cann_versions:
        lines.append(f"  CANN versions: {info.cann_versions}")
    if info.references:
        lines.append(f"  References:    {info.references}")
    return "\n".join(lines)


def get_all_codes() -> Dict[str, ErrorInfo]:
    """Return the complete error code database."""
    return dict(_ERROR_DB)


def search_errors(keyword: str) -> List[ErrorInfo]:
    """Search error database by keyword.

    Args:
        keyword: Search term (case-insensitive).

    Returns:
        List of matching :class:`ErrorInfo` items.
    """
    keyword = keyword.lower()
    return [
        info for info in _ERROR_DB.values()
        if keyword in info.summary.lower()
        or keyword in info.likely_cause.lower()
        or keyword in info.fix.lower()
        or keyword in info.category.lower()
    ]
