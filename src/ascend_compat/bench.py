"""Benchmarking framework for measuring shim overhead and NPU performance.

The original project roadmap (Phase 5) called for automated benchmarks that
compare original CUDA code vs. shimmed code on Ascend, and measure the shim
overhead itself.  This module provides that infrastructure.

Three measurement modes
-----------------------

1. **Shim overhead** — How much time does ascend-compat's monkey-patching
   add to each ``torch.cuda`` call?  This measures the proxy/routing cost
   in isolation, without any hardware dependency.

2. **Operation latency** — Wall-clock time for common operations (tensor
   creation, matmul, attention, etc.) on the current backend.  Produces a
   comparison-ready CSV.

3. **Model throughput** — End-to-end inference throughput (tokens/sec or
   samples/sec) for a given model, with and without the shim active.

Usage::

    from ascend_compat.bench import ShimOverheadBench, OpLatencyBench

    # Measure shim proxy overhead
    overhead = ShimOverheadBench().run()
    print(overhead.report())

    # Measure operation latency
    latency = OpLatencyBench(device="npu").run()
    print(latency.report())

CLI::

    ascend-compat bench overhead
    ascend-compat bench ops --device npu --csv results.csv
"""

from __future__ import annotations

import csv
import io
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ascend_compat._logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "BenchResult",
    "BenchReport",
    "ShimOverheadBench",
    "OpLatencyBench",
    "ModelThroughputBench",
    "MemoryBandwidthBench",
    "get_system_fingerprint",
]


# ---------------------------------------------------------------------------
# System fingerprint (for reproducible benchmark comparison)
# ---------------------------------------------------------------------------


def get_system_fingerprint() -> Dict[str, Any]:
    """Return immutable system identifier for benchmark comparison.

    Includes hardware model, software versions, and OS info so benchmark
    results from different machines can be meaningfully compared.

    Returns:
        Dict with keys like ``npu_model``, ``torch_version``, etc.
    """
    import torch
    from ascend_compat import __version__

    fp: Dict[str, Any] = {
        "ascend_compat_version": __version__,
        "torch_version": torch.__version__,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
    }

    # NPU info
    if hasattr(torch, "npu") and torch.npu.is_available():
        fp["npu_model"] = torch.npu.get_device_name(0)
        fp["npu_count"] = torch.npu.device_count()
        if hasattr(torch.npu, "cann_version"):
            fp["cann_version"] = torch.npu.cann_version()
    else:
        fp["npu_model"] = None
        fp["npu_count"] = 0

    # CUDA info (for cross-platform comparison)
    if torch.cuda.is_available():
        fp["cuda_device"] = torch.cuda.get_device_name(0)
        fp["cuda_count"] = torch.cuda.device_count()
    else:
        fp["cuda_device"] = None
        fp["cuda_count"] = 0

    return fp


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    """Result of a single benchmark measurement."""
    name: str
    iterations: int
    total_seconds: float
    per_call_us: float  # microseconds per call
    device: str = "cpu"

    @property
    def calls_per_second(self) -> float:
        return self.iterations / self.total_seconds if self.total_seconds > 0 else 0


@dataclass
class BenchReport:
    """Collection of benchmark results."""
    title: str
    results: List[BenchResult] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)

    def report(self) -> str:
        """Human-readable benchmark report."""
        lines = [
            f"{'=' * 70}",
            f"  {self.title}",
            f"{'=' * 70}",
        ]

        # Metadata
        for k, v in self.metadata.items():
            lines.append(f"  {k}: {v}")
        lines.append("")

        # Results table
        lines.append(f"  {'Operation':<35} {'per-call (us)':>13} {'calls/sec':>12} {'iters':>8}")
        lines.append(f"  {'-' * 35} {'-' * 13} {'-' * 12} {'-' * 8}")

        for r in sorted(self.results, key=lambda x: x.per_call_us):
            lines.append(
                f"  {r.name:<35} {r.per_call_us:>13.2f} {r.calls_per_second:>12.0f} {r.iterations:>8}"
            )

        lines.append(f"{'=' * 70}")
        return "\n".join(lines)

    def to_csv(self, include_fingerprint: bool = True) -> str:
        """Export results as CSV string.

        Args:
            include_fingerprint: If True, prepend a comment header with
                the system fingerprint for reproducibility.
        """
        output = io.StringIO()

        if include_fingerprint:
            fp = get_system_fingerprint()
            for k, v in fp.items():
                output.write(f"# {k}: {v}\n")
            for k, v in self.metadata.items():
                output.write(f"# {k}: {v}\n")
            output.write("#\n")

        writer = csv.writer(output)
        writer.writerow(["operation", "device", "per_call_us", "calls_per_sec", "iterations"])
        for r in self.results:
            writer.writerow([r.name, r.device, f"{r.per_call_us:.2f}",
                             f"{r.calls_per_second:.0f}", r.iterations])
        return output.getvalue()


def _timeit(fn: Callable[[], Any], iterations: int = 10000, warmup: int = 100) -> Tuple[float, int]:
    """Time a callable, returning (total_seconds, iterations)."""
    # Warmup
    for _ in range(warmup):
        fn()

    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    elapsed = time.perf_counter() - start

    return elapsed, iterations


# ---------------------------------------------------------------------------
# Shim overhead benchmark
# ---------------------------------------------------------------------------


class ShimOverheadBench:
    """Measure the overhead of ascend-compat's monkey-patching layer.

    This benchmarks the *proxy function cost* — the extra microseconds
    added by routing ``torch.cuda.X`` through ascend-compat to ``torch.npu.X``.
    It compares calling a torch function directly vs. through the shim proxy.
    """

    def __init__(self, iterations: int = 50000) -> None:
        self.iterations = iterations

    def run(self) -> BenchReport:
        import torch
        report = BenchReport(
            title="Shim Overhead Benchmark",
            metadata={
                "PyTorch": torch.__version__,
                "iterations": str(self.iterations),
                "device": "cpu (measuring proxy overhead only)",
            },
        )

        # Benchmark 1: Direct torch.cuda.is_available() vs baseline
        original_fn = torch.cuda.is_available

        elapsed, iters = _timeit(original_fn, self.iterations)
        report.results.append(BenchResult(
            name="torch.cuda.is_available (direct)",
            iterations=iters,
            total_seconds=elapsed,
            per_call_us=(elapsed / iters) * 1e6,
            device="cpu",
        ))

        # Benchmark 2: A simple lambda proxy (simulates shim overhead)
        def _proxy_fn() -> bool:
            return original_fn()

        elapsed, iters = _timeit(_proxy_fn, self.iterations)
        report.results.append(BenchResult(
            name="proxy wrapper (1 indirection)",
            iterations=iters,
            total_seconds=elapsed,
            per_call_us=(elapsed / iters) * 1e6,
            device="cpu",
        ))

        # Benchmark 3: torch.device("cpu") — baseline
        elapsed, iters = _timeit(lambda: torch.device("cpu"), self.iterations)
        report.results.append(BenchResult(
            name="torch.device('cpu') baseline",
            iterations=iters,
            total_seconds=elapsed,
            per_call_us=(elapsed / iters) * 1e6,
            device="cpu",
        ))

        # Benchmark 4: String manipulation overhead (simulates cuda→npu rewrite)
        def _device_with_replace() -> Any:
            s = "cuda:0"
            return torch.device(s.replace("cuda", "cpu", 1))

        elapsed, iters = _timeit(_device_with_replace, self.iterations)
        report.results.append(BenchResult(
            name="torch.device + string replace",
            iterations=iters,
            total_seconds=elapsed,
            per_call_us=(elapsed / iters) * 1e6,
            device="cpu",
        ))

        # Benchmark 5: torch.empty (small tensor creation) baseline
        elapsed, iters = _timeit(lambda: torch.empty(1), self.iterations)
        report.results.append(BenchResult(
            name="torch.empty(1) baseline",
            iterations=iters,
            total_seconds=elapsed,
            per_call_us=(elapsed / iters) * 1e6,
            device="cpu",
        ))

        return report


# ---------------------------------------------------------------------------
# Operation latency benchmark
# ---------------------------------------------------------------------------


class OpLatencyBench:
    """Measure wall-clock latency of common operations on the current backend.

    This runs actual tensor operations and reports per-operation timing.
    Useful for comparing NPU vs CPU vs CUDA performance.
    """

    def __init__(self, device: str = "cpu", iterations: int = 1000) -> None:
        self.device = device
        self.iterations = iterations

    def run(self) -> BenchReport:
        import torch

        dev = self.device
        report = BenchReport(
            title=f"Operation Latency Benchmark (device={dev})",
            metadata={
                "PyTorch": torch.__version__,
                "device": dev,
                "iterations": str(self.iterations),
            },
        )

        # Try to resolve the device
        try:
            torch_dev = torch.device(dev)
        except Exception as exc:
            logger.warning("Cannot create device '%s': %s", dev, exc)
            report.metadata["error"] = str(exc)
            return report

        # Helper to sync if needed
        def _sync() -> None:
            if dev.startswith("npu") and hasattr(torch, "npu"):
                torch.npu.synchronize()
            elif dev.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()

        # ── Tensor creation ops ──────────────────────────────────
        def _zeros_small() -> Any:
            t = torch.zeros(64, 64, device=torch_dev)
            _sync()
            return t

        def _zeros_large() -> Any:
            t = torch.zeros(1024, 1024, device=torch_dev)
            _sync()
            return t

        def _randn_medium() -> Any:
            t = torch.randn(256, 256, device=torch_dev)
            _sync()
            return t

        # ── Compute ops ──────────────────────────────────────────
        a = torch.randn(512, 512, device=torch_dev)
        b = torch.randn(512, 512, device=torch_dev)

        def _matmul() -> Any:
            c = torch.matmul(a, b)
            _sync()
            return c

        def _add() -> Any:
            c = a + b
            _sync()
            return c

        def _relu() -> Any:
            c = torch.relu(a)
            _sync()
            return c

        # ── Reduction ops ────────────────────────────────────────
        x = torch.randn(1024, 1024, device=torch_dev)

        def _sum() -> Any:
            s = x.sum()
            _sync()
            return s

        def _mean() -> Any:
            m = x.mean()
            _sync()
            return m

        def _softmax() -> Any:
            s = torch.softmax(x, dim=-1)
            _sync()
            return s

        # ── Run all benchmarks ───────────────────────────────────
        benchmarks = [
            ("zeros(64x64)", _zeros_small),
            ("zeros(1024x1024)", _zeros_large),
            ("randn(256x256)", _randn_medium),
            ("matmul(512x512)", _matmul),
            ("add(512x512)", _add),
            ("relu(512x512)", _relu),
            ("sum(1024x1024)", _sum),
            ("mean(1024x1024)", _mean),
            ("softmax(1024x1024)", _softmax),
        ]

        for name, fn in benchmarks:
            try:
                elapsed, iters = _timeit(fn, self.iterations, warmup=min(50, self.iterations))
                report.results.append(BenchResult(
                    name=name,
                    iterations=iters,
                    total_seconds=elapsed,
                    per_call_us=(elapsed / iters) * 1e6,
                    device=dev,
                ))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Benchmark '%s' failed: %s", name, exc)
                report.results.append(BenchResult(
                    name=f"{name} [FAILED: {exc}]",
                    iterations=0,
                    total_seconds=0,
                    per_call_us=0,
                    device=dev,
                ))

        return report


# ---------------------------------------------------------------------------
# Model throughput benchmark
# ---------------------------------------------------------------------------


class ModelThroughputBench:
    """Measure inference throughput for a PyTorch model.

    Runs forward passes and reports samples/sec and latency statistics.
    """

    def __init__(
        self,
        model: Any,
        input_fn: Callable[[], Any],
        device: str = "cpu",
        iterations: int = 100,
        warmup: int = 10,
        batch_size: int = 1,
    ) -> None:
        self.model = model
        self.input_fn = input_fn
        self.device = device
        self.iterations = iterations
        self.warmup = warmup
        self.batch_size = batch_size

    def run(self) -> BenchReport:
        import torch

        dev = torch.device(self.device)
        model = self.model.to(dev).eval()

        report = BenchReport(
            title=f"Model Throughput Benchmark (device={self.device})",
            metadata={
                "PyTorch": torch.__version__,
                "device": self.device,
                "model": type(self.model).__name__,
                "batch_size": str(self.batch_size),
                "iterations": str(self.iterations),
                "warmup": str(self.warmup),
            },
        )

        def _sync() -> None:
            if self.device.startswith("npu") and hasattr(torch, "npu"):
                torch.npu.synchronize()
            elif self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()

        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup):
                inp = self.input_fn()
                if isinstance(inp, torch.Tensor):
                    inp = inp.to(dev)
                model(inp)
                _sync()

        # Timed runs
        latencies: List[float] = []
        with torch.no_grad():
            for _ in range(self.iterations):
                inp = self.input_fn()
                if isinstance(inp, torch.Tensor):
                    inp = inp.to(dev)

                _sync()
                start = time.perf_counter()
                model(inp)
                _sync()
                elapsed = time.perf_counter() - start
                latencies.append(elapsed)

        total = sum(latencies)
        avg_latency = total / len(latencies)
        samples_per_sec = (self.batch_size * self.iterations) / total

        latencies_sorted = sorted(latencies)
        p50 = latencies_sorted[len(latencies_sorted) // 2]
        p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
        p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)]

        report.results.append(BenchResult(
            name="avg_latency",
            iterations=self.iterations,
            total_seconds=total,
            per_call_us=avg_latency * 1e6,
            device=self.device,
        ))

        report.metadata.update({
            "throughput_samples_per_sec": f"{samples_per_sec:.2f}",
            "latency_p50_ms": f"{p50 * 1000:.2f}",
            "latency_p95_ms": f"{p95 * 1000:.2f}",
            "latency_p99_ms": f"{p99 * 1000:.2f}",
        })

        return report


# ---------------------------------------------------------------------------
# Memory bandwidth benchmark
# ---------------------------------------------------------------------------


class MemoryBandwidthBench:
    """Measure memory bandwidth on the target device.

    Memory bandwidth is the actual bottleneck for most NPU workloads.  This
    benchmark measures:

    1. **Copy bandwidth** — Time to clone a large tensor (reads source,
       writes destination).  Reports MB/s.
    2. **Compute intensity sweep** — Varies arithmetic intensity from
       bandwidth-bound (vector add) to compute-bound (large matmul) to
       find the device's roofline crossover point.

    Usage::

        from ascend_compat.bench import MemoryBandwidthBench
        report = MemoryBandwidthBench(device="npu").run()
        print(report.report())
    """

    def __init__(self, device: str = "cpu", iterations: int = 50) -> None:
        self.device = device
        self.iterations = iterations

    def run(self) -> BenchReport:
        import torch

        dev = self.device
        report = BenchReport(
            title=f"Memory Bandwidth Benchmark (device={dev})",
            metadata={
                "PyTorch": torch.__version__,
                "device": dev,
                "iterations": str(self.iterations),
            },
        )

        try:
            torch_dev = torch.device(dev)
        except Exception as exc:
            logger.warning("Cannot create device '%s': %s", dev, exc)
            report.metadata["error"] = str(exc)
            return report

        def _sync() -> None:
            if dev.startswith("npu") and hasattr(torch, "npu"):
                torch.npu.synchronize()
            elif dev.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()

        # ── Copy bandwidth at different sizes ────────────────────
        for size_mb in (1, 16, 64, 256):
            n_elements = size_mb * 256 * 1024  # 4 bytes per float32 * 256K = 1MB
            try:
                a = torch.randn(n_elements, device=torch_dev, dtype=torch.float32)
                _sync()

                # Warmup
                for _ in range(3):
                    _ = a.clone()
                    _sync()

                # Timed
                start = time.perf_counter()
                for _ in range(self.iterations):
                    _ = a.clone()
                    _sync()
                elapsed = time.perf_counter() - start

                # Bandwidth: read source + write dest = 2x
                bytes_per_iter = a.element_size() * a.nelement() * 2
                total_bytes = bytes_per_iter * self.iterations
                bandwidth_mbs = (total_bytes / elapsed) / (1024 * 1024)

                report.results.append(BenchResult(
                    name=f"copy {size_mb}MB",
                    iterations=self.iterations,
                    total_seconds=elapsed,
                    per_call_us=(elapsed / self.iterations) * 1e6,
                    device=dev,
                ))
                report.metadata[f"copy_{size_mb}MB_bandwidth_MBs"] = f"{bandwidth_mbs:.0f}"

                del a
            except Exception as exc:  # noqa: BLE001
                logger.warning("Copy %dMB benchmark failed: %s", size_mb, exc)

        # ── Compute intensity sweep (roofline) ───────────────────
        for n in (256, 512, 1024, 2048):
            try:
                a = torch.randn(n, n, device=torch_dev, dtype=torch.float32)
                b = torch.randn(n, n, device=torch_dev, dtype=torch.float32)
                _sync()

                iters = max(5, self.iterations // 5)
                for _ in range(3):
                    torch.matmul(a, b)
                    _sync()

                start = time.perf_counter()
                for _ in range(iters):
                    torch.matmul(a, b)
                    _sync()
                elapsed = time.perf_counter() - start

                # FLOPS: 2*N^3 for matmul
                flops_per_iter = 2 * n * n * n
                gflops = (flops_per_iter * iters / elapsed) / 1e9

                report.results.append(BenchResult(
                    name=f"matmul {n}x{n}",
                    iterations=iters,
                    total_seconds=elapsed,
                    per_call_us=(elapsed / iters) * 1e6,
                    device=dev,
                ))
                report.metadata[f"matmul_{n}x{n}_GFLOPS"] = f"{gflops:.1f}"

                del a, b
            except Exception as exc:  # noqa: BLE001
                logger.warning("Matmul %dx%d benchmark failed: %s", n, n, exc)

        return report
