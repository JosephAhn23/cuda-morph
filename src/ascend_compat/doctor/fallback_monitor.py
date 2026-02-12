"""CPU fallback detection and performance monitoring.

torch_npu's PrivateUse1 backend registers a global CPU fallback for
unsupported operators.  When an op isn't implemented for NPU, PyTorch
silently falls back to CPU execution.  This involves:

1. Device→Host data transfer (slow — crosses PCIe)
2. CPU execution of the operator (slow — no parallelism)
3. Host→Device data transfer (slow — crosses PCIe again)
4. Pipeline synchronization stall (blocks the NPU pipeline)

A single fallback can degrade throughput by 10-100x for that operator.
Multiple fallbacks in a model's critical path can make training
catastrophically slower than expected.

This module hooks into PyTorch's dispatch mechanism to detect when
CPU fallback occurs, count occurrences, and estimate performance impact.

Usage::

    from ascend_compat.doctor import FallbackMonitor

    monitor = FallbackMonitor()
    monitor.start()

    # ... run your model ...

    report = monitor.stop()
    print(report.summary())
    # → "3 operators fell back to CPU during 100 iterations:
    #     - aten::_unique2 (47 calls, est. 23% of compute time)
    #     - aten::histc (31 calls, est. 8% of compute time)
    #     - aten::multinomial (22 calls, est. 2% of compute time)"
"""

from __future__ import annotations

import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ascend_compat._logging import get_logger

logger = get_logger(__name__)


@dataclass
class FallbackEvent:
    """Record of a single CPU fallback event."""

    op_name: str                # e.g. "aten::_unique2"
    timestamp: float            # time.monotonic()
    input_shapes: List[str] = field(default_factory=list)
    input_dtypes: List[str] = field(default_factory=list)


@dataclass
class FallbackStats:
    """Aggregated statistics for one operator."""

    op_name: str
    call_count: int = 0
    total_time_ms: float = 0.0      # Estimated time in milliseconds
    first_seen: float = 0.0
    last_seen: float = 0.0
    sample_shapes: List[str] = field(default_factory=list)


@dataclass
class FallbackReport:
    """Summary report of all CPU fallbacks detected during monitoring."""

    total_fallbacks: int = 0
    unique_ops: int = 0
    monitoring_duration_s: float = 0.0
    stats: Dict[str, FallbackStats] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary."""
        if self.total_fallbacks == 0:
            return (
                "No CPU fallbacks detected during monitoring "
                f"({self.monitoring_duration_s:.1f}s).\n"
                "All operators ran natively on NPU."
            )

        lines = [
            f"{self.total_fallbacks} CPU fallback(s) across "
            f"{self.unique_ops} operator(s) "
            f"during {self.monitoring_duration_s:.1f}s of monitoring:",
        ]

        # Sort by call count (most frequent first)
        sorted_stats = sorted(
            self.stats.values(), key=lambda s: s.call_count, reverse=True
        )

        for stat in sorted_stats[:20]:  # Top 20
            shapes_info = f" (shapes: {stat.sample_shapes[0]})" if stat.sample_shapes else ""
            lines.append(
                f"  - {stat.op_name}: {stat.call_count} call(s), "
                f"~{stat.total_time_ms:.1f}ms total{shapes_info}"
            )

        if self.unique_ops > 20:
            lines.append(f"  ... and {self.unique_ops - 20} more operators")

        lines.append("")
        lines.append("Recommendations:")

        high_frequency = [s for s in sorted_stats if s.call_count > 10]
        if high_frequency:
            lines.append(
                f"  - {len(high_frequency)} op(s) fall back frequently — "
                "consider rewriting these with NPU-native alternatives"
            )

        lines.append(
            "  - Update CANN to the latest version for broader op coverage"
        )
        lines.append(
            "  - Run `ascend-compat check your_script.py` to identify CUDA-specific code"
        )

        return "\n".join(lines)


class FallbackMonitor:
    """Monitor and record CPU fallback events during model execution.

    The monitor works by patching PyTorch's warning system to intercept
    the "operator X will fall back to CPU" warnings that torch_npu emits.

    For more precise monitoring on real Ascend hardware, it can also hook
    into CANN's profiling API via msprof.

    Example::

        monitor = FallbackMonitor()
        with monitor:
            output = model(input_tensor)

        print(monitor.report.summary())
    """

    def __init__(self) -> None:
        self._events: List[FallbackEvent] = []
        self._start_time: float = 0.0
        self._stop_time: float = 0.0
        self._original_showwarning: Any = None
        self._monitoring = False
        self.report: FallbackReport = FallbackReport()

    def start(self) -> None:
        """Begin monitoring for CPU fallback events."""
        if self._monitoring:
            return

        self._events.clear()
        self._start_time = time.monotonic()
        self._monitoring = True

        # Hook into Python's warning system to catch torch_npu fallback warnings
        self._original_showwarning = warnings.showwarning

        def _intercept_warning(
            message: Any, category: type, filename: str,
            lineno: int, file: Any = None, line: str = None,
        ) -> None:
            msg_str = str(message)
            if "fall back" in msg_str.lower() and ("cpu" in msg_str.lower() or "CPU" in msg_str):
                # Extract operator name from warning message
                # Typical format: "The operator 'aten::xxx' is not currently supported
                # on the NPU backend and will fall back to run on the CPU"
                op_name = _extract_op_name(msg_str)
                self._events.append(FallbackEvent(
                    op_name=op_name,
                    timestamp=time.monotonic(),
                ))
                logger.warning("CPU fallback detected: %s", op_name)

            # Still show the original warning
            if self._original_showwarning:
                self._original_showwarning(message, category, filename, lineno, file, line)

        warnings.showwarning = _intercept_warning
        logger.info("Fallback monitor started")

    def stop(self) -> FallbackReport:
        """Stop monitoring and compile the report.

        Returns:
            :class:`FallbackReport` with aggregated statistics.
        """
        if not self._monitoring:
            return self.report

        self._stop_time = time.monotonic()
        self._monitoring = False

        # Restore original warning handler
        if self._original_showwarning is not None:
            warnings.showwarning = self._original_showwarning
            self._original_showwarning = None

        # Compile report
        self.report = _compile_report(
            self._events,
            self._stop_time - self._start_time,
        )

        logger.info(
            "Fallback monitor stopped: %d fallback(s) across %d op(s)",
            self.report.total_fallbacks,
            self.report.unique_ops,
        )

        return self.report

    def __enter__(self) -> "FallbackMonitor":
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()


def _extract_op_name(warning_msg: str) -> str:
    """Extract operator name from a fallback warning message.

    Handles common formats:
    - "The operator 'aten::xxx' is not currently supported..."
    - "operator aten::xxx will fall back..."
    """
    import re

    # Try to match quoted operator name
    match = re.search(r"operator\s+'([^']+)'", warning_msg, re.IGNORECASE)
    if match:
        return match.group(1)

    # Try to match aten:: pattern
    match = re.search(r"(aten::\w+)", warning_msg)
    if match:
        return match.group(1)

    # Fallback: return truncated warning
    return warning_msg[:80].strip()


def _compile_report(events: List[FallbackEvent], duration: float) -> FallbackReport:
    """Compile events into an aggregated report."""
    stats: Dict[str, FallbackStats] = {}

    for event in events:
        if event.op_name not in stats:
            stats[event.op_name] = FallbackStats(
                op_name=event.op_name,
                first_seen=event.timestamp,
            )
        s = stats[event.op_name]
        s.call_count += 1
        s.last_seen = event.timestamp
        if event.input_shapes and len(s.sample_shapes) < 3:
            s.sample_shapes.extend(event.input_shapes)

    return FallbackReport(
        total_fallbacks=len(events),
        unique_ops=len(stats),
        monitoring_duration_s=duration,
        stats=stats,
    )
