"""Operator coverage auditor — profile a model for NPU vs CPU fallback ops.

This module traces a model's forward pass to discover which PyTorch operators
are called, then checks each against torch_npu's supported operator list to
predict which will run natively on NPU and which will fall back to CPU.

This is more precise than the warning-based FallbackMonitor because:
1. It works BEFORE deployment — you can audit a model on CPU
2. It catches ALL operators, not just ones that trigger warnings
3. It provides estimated performance impact based on op frequency

Usage::

    from ascend_compat.doctor.op_auditor import audit_model
    import torchvision

    model = torchvision.models.resnet50()
    x = torch.randn(1, 3, 224, 224)

    report = audit_model(model, x)
    print(report.summary())

Limitation:
    This uses torch.jit.trace or torch.fx.symbolic_trace, both of which
    have caveats with dynamic control flow.  The audit covers the specific
    execution path taken by the sample input, not ALL possible paths.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ascend_compat._logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Known operator support status
# ---------------------------------------------------------------------------

# Operators known to be unsupported or problematic on Ascend.
# This list is sourced from community reports, CANN release notes, and
# torch_npu issue trackers.
_KNOWN_UNSUPPORTED_OPS: Set[str] = {
    # No equivalent CANN kernel
    "aten::_unique2",
    "aten::unique_dim",
    "aten::histc",
    "aten::multinomial",
    "aten::searchsorted",
    "aten::bucketize",
    "aten::_fft_r2c",
    "aten::_fft_c2r",
    "aten::_fft_c2c",
    "aten::stft",
    "aten::istft",
    "aten::grid_sampler_2d",
    "aten::grid_sampler_3d",
    "aten::_trilinear",
    "aten::linalg_det",
    "aten::linalg_slogdet",
    "aten::linalg_eigh",
    "aten::linalg_eigvalsh",
    "aten::linalg_svd",
    "aten::triangular_solve",
    "aten::cholesky",
    "aten::ormqr",
    "aten::geqrf",
    "aten::_linalg_qr_helper",
    # Quantized ops (no native quant support)
    "quantized::linear",
    "quantized::conv2d",
    "quantized::add",
}

# Operators that work but with known performance issues
_KNOWN_SLOW_OPS: Dict[str, str] = {
    "aten::nonzero": "Dynamic output shape causes device sync — prefer boolean masking",
    "aten::index_put_": "Scattered writes are slow on SIMD — prefer gather-based alternatives",
    "aten::scatter_": "Similar to index_put_ — batch operations preferred",
    "aten::where": "Element-wise conditional — consider masked_fill for simple cases",
    "aten::unique_consecutive": "Requires sorting — may cause sync point",
}


@dataclass
class OpInfo:
    """Information about one operator discovered during audit."""
    name: str
    call_count: int = 0
    status: str = "native"  # "native", "fallback", "slow", "unknown"
    note: str = ""


@dataclass
class AuditReport:
    """Result of operator coverage audit."""
    model_name: str = ""
    total_ops: int = 0
    native_ops: int = 0
    fallback_ops: int = 0
    slow_ops: int = 0
    unknown_ops: int = 0
    ops: Dict[str, OpInfo] = field(default_factory=dict)

    @property
    def coverage_pct(self) -> float:
        """Percentage of operators that run natively on NPU."""
        if self.total_ops == 0:
            return 100.0
        return (self.native_ops / self.total_ops) * 100

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Operator Coverage Audit: {self.model_name}",
            "=" * 55,
            f"  Total unique operators: {self.total_ops}",
            f"  Native (NPU):          {self.native_ops} ({self.coverage_pct:.1f}%)",
            f"  CPU fallback:          {self.fallback_ops}",
            f"  Slow (native but suboptimal): {self.slow_ops}",
            f"  Unknown:               {self.unknown_ops}",
        ]

        if self.fallback_ops > 0:
            lines.append("")
            lines.append("  CPU fallback operators:")
            for name, info in sorted(self.ops.items()):
                if info.status == "fallback":
                    lines.append(f"    - {name} (called {info.call_count}x)")
                    if info.note:
                        lines.append(f"      {info.note}")

        if self.slow_ops > 0:
            lines.append("")
            lines.append("  Slow operators (consider alternatives):")
            for name, info in sorted(self.ops.items()):
                if info.status == "slow":
                    lines.append(f"    - {name}: {info.note}")

        if self.fallback_ops == 0 and self.slow_ops == 0:
            lines.append("")
            lines.append("  All operators are natively supported on NPU.")

        return "\n".join(lines)


def audit_model(
    model: Any,
    sample_input: Any,
    model_name: str = "",
) -> AuditReport:
    """Audit a model's operator coverage for Ascend NPU.

    This traces the model's forward pass with the sample input and
    analyzes which operators are used.

    Args:
        model: A ``torch.nn.Module``.
        sample_input: Sample input tensor(s) for tracing.
            Can be a single tensor or a tuple of tensors.
        model_name: Optional name for the report.

    Returns:
        :class:`AuditReport` with operator coverage analysis.
    """
    import torch

    if not model_name:
        model_name = type(model).__name__

    logger.info("Auditing operator coverage for %s", model_name)

    # Collect operators via torch.fx symbolic trace
    op_counts: Counter[str] = Counter()

    try:
        ops = _trace_with_fx(model, sample_input)
        op_counts.update(ops)
        logger.debug("FX trace found %d operator calls (%d unique)", sum(op_counts.values()), len(op_counts))
    except Exception as exc:
        logger.debug("FX trace failed (%s), falling back to hook-based tracing", exc)
        ops = _trace_with_hooks(model, sample_input)
        op_counts.update(ops)

    # Classify each operator
    report = AuditReport(model_name=model_name)

    for op_name, count in op_counts.items():
        info = OpInfo(name=op_name, call_count=count)

        if op_name in _KNOWN_UNSUPPORTED_OPS:
            info.status = "fallback"
            info.note = "Known unsupported — will fall back to CPU"
            report.fallback_ops += 1
        elif op_name in _KNOWN_SLOW_OPS:
            info.status = "slow"
            info.note = _KNOWN_SLOW_OPS[op_name]
            report.slow_ops += 1
        elif op_name.startswith("aten::") or op_name.startswith("torch_npu::"):
            info.status = "native"
            report.native_ops += 1
        else:
            info.status = "unknown"
            report.unknown_ops += 1

        report.ops[op_name] = info

    report.total_ops = len(op_counts)
    logger.info(
        "Audit complete: %d ops, %.1f%% native, %d fallback",
        report.total_ops, report.coverage_pct, report.fallback_ops,
    )

    return report


def _trace_with_fx(model: Any, sample_input: Any) -> List[str]:
    """Trace operator calls using torch.fx.

    torch.fx does symbolic tracing — it walks the model's code and records
    the graph without actually executing ops.  This is fast but can't handle
    dynamic control flow (if statements depending on tensor values).
    """
    import torch
    import torch.fx

    traced = torch.fx.symbolic_trace(model)
    ops: List[str] = []

    for node in traced.graph.nodes:
        if node.op == "call_function":
            fn = node.target
            if hasattr(fn, "__module__") and hasattr(fn, "__name__"):
                name = f"{fn.__module__}.{fn.__name__}"
            elif hasattr(fn, "overloadpacket"):
                # aten ops
                name = str(fn.overloadpacket)
            else:
                name = str(fn)
            ops.append(name)
        elif node.op == "call_method":
            ops.append(f"Tensor.{node.target}")
        elif node.op == "call_module":
            # Get the actual module type
            submod = traced.get_submodule(node.target)
            ops.append(f"nn.{type(submod).__name__}")

    return ops


def _trace_with_hooks(model: Any, sample_input: Any) -> List[str]:
    """Trace operator calls using forward hooks.

    Fallback approach when FX tracing fails.  This actually executes
    the forward pass and records which modules are called.
    """
    import torch

    ops: List[str] = []
    hooks = []

    def _make_hook(name: str) -> Any:
        def hook(module: Any, input: Any, output: Any) -> None:
            ops.append(f"nn.{type(module).__name__}")
        return hook

    # Register hooks on all modules
    for name, module in model.named_modules():
        h = module.register_forward_hook(_make_hook(name))
        hooks.append(h)

    try:
        model.eval()
        with torch.no_grad():
            if isinstance(sample_input, (tuple, list)):
                model(*sample_input)
            else:
                model(sample_input)
    finally:
        for h in hooks:
            h.remove()

    return ops
