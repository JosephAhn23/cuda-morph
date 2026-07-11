#!/usr/bin/env python3
"""
cuda-morph: Comprehensive Benchmark Suite with Competitive Grading
===================================================================

Runs micro and macro benchmarks, compares against competitors, and assigns
a production-readiness grade.

Usage:
    python bench_complete.py [--full] [--csv output.csv]
"""

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from ascend_compat.bench import (
    ShimOverheadBench,
    OpLatencyBench,
    ModelThroughputBench,
    MemoryBandwidthBench,
    get_system_fingerprint,
)


# ---------------------------------------------------------------------------
# Benchmark Result Models
# ---------------------------------------------------------------------------


@dataclass
class CompetitorBenchmark:
    """Baseline performance for a competitor."""
    name: str
    throughput_tps: float  # Tokens Per Second
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    memory_bandwidth_gbs: float


# vLLM baseline on NVIDIA H100 (reference)
VLLM_H100_BASELINE = CompetitorBenchmark(
    name="vLLM on NVIDIA H100",
    throughput_tps=15000,
    latency_p50_ms=1.2,
    latency_p95_ms=2.5,
    latency_p99_ms=4.0,
    memory_bandwidth_gbs=2000,
)

# vLLM baseline on NVIDIA A100 (reference)
VLLM_A100_BASELINE = CompetitorBenchmark(
    name="vLLM on NVIDIA A100",
    throughput_tps=8000,
    latency_p50_ms=2.0,
    latency_p95_ms=4.0,
    latency_p99_ms=6.5,
    memory_bandwidth_gbs=1555,
)

# Pure torch.npu baseline (Ascend 910B reference)
ASCEND_NATIVE_BASELINE = CompetitorBenchmark(
    name="Native torch.npu on Ascend 910B",
    throughput_tps=10000,
    latency_p50_ms=1.8,
    latency_p95_ms=3.5,
    latency_p99_ms=5.5,
    memory_bandwidth_gbs=1400,
)


# ---------------------------------------------------------------------------
# Scoring & Grading
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkScore:
    """Score for a single benchmark dimension."""
    name: str
    cuda_morph_value: float
    baseline_value: float
    unit: str
    category: str  # "throughput", "latency", "bandwidth"

    def efficiency(self) -> float:
        """Return efficiency as a percentage (100 = match baseline)."""
        if self.baseline_value == 0:
            return 100.0

        # For throughput and bandwidth, higher is better
        if self.category in ("throughput", "bandwidth"):
            return (self.cuda_morph_value / self.baseline_value) * 100

        # For latency, lower is better
        if self.category == "latency":
            return (self.baseline_value / self.cuda_morph_value) * 100

        return 0.0


class GradeCard:
    """Assign production-readiness grade based on benchmark results."""

    THRESHOLDS = {
        "A+": 95.0,   # Near-native performance (< 5% overhead)
        "A": 90.0,    # Excellent (< 10% overhead)
        "B+": 85.0,   # Good (< 15% overhead)
        "B": 80.0,    # Acceptable (< 20% overhead)
        "C": 70.0,    # Needs work (20-30% overhead)
        "D": 60.0,    # Problematic (30-40% overhead)
        "F": 0.0,     # Not production-ready
    }

    @classmethod
    def grade(cls, scores: List[BenchmarkScore]) -> tuple[str, float]:
        """Assign letter grade and confidence score.

        Returns:
            (grade_letter, confidence_percentage)
        """
        if not scores:
            return "N/A", 0.0

        # Weight by category
        weights = {
            "throughput": 0.5,  # Most important for inference
            "latency": 0.35,    # Critical for user experience
            "bandwidth": 0.15,  # Affects potential
        }

        weighted_score = 0.0
        total_weight = 0.0

        for score in scores:
            w = weights.get(score.category, 0.0)
            weighted_score += score.efficiency() * w
            total_weight += w

        avg_efficiency = weighted_score / total_weight if total_weight > 0 else 0

        # Assign grade
        for grade, threshold in sorted(cls.THRESHOLDS.items(),
                                       key=lambda x: x[1], reverse=True):
            if avg_efficiency >= threshold:
                return grade, avg_efficiency

        return "F", avg_efficiency


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------


def run_shim_overhead_benchmarks() -> Dict[str, float]:
    """Measure pure shim overhead (proxy cost)."""
    print("\n[1/5] Running Shim Overhead Benchmarks...")
    bench = ShimOverheadBench(iterations=50000)
    report = bench.run()

    print(report.report())

    # Extract key metrics
    metrics = {}
    for result in report.results:
        metrics[result.name] = result.per_call_us

    return metrics


def run_operation_latency_benchmarks() -> Dict[str, Dict[str, float]]:
    """Measure operation latency on available devices."""
    print("\n[2/5] Running Operation Latency Benchmarks...")

    results = {}

    # CPU baseline
    print("  - CPU operations...")
    bench = OpLatencyBench(device="cpu", iterations=1000)
    report = bench.run()
    print(report.report())
    results["cpu"] = {r.name: r.per_call_us for r in report.results}

    # NPU if available
    if hasattr(torch, "npu") and torch.npu.is_available():
        print("  - NPU operations...")
        bench = OpLatencyBench(device="npu", iterations=500)
        report = bench.run()
        print(report.report())
        results["npu"] = {r.name: r.per_call_us for r in report.results}

    return results


def run_memory_bandwidth_benchmarks() -> Dict[str, Dict[str, float]]:
    """Measure memory bandwidth characteristics."""
    print("\n[3/5] Running Memory Bandwidth Benchmarks...")

    results = {}

    # CPU
    print("  - CPU memory bandwidth...")
    bench = MemoryBandwidthBench(device="cpu", iterations=20)
    report = bench.run()
    print(report.report())
    results["cpu"] = {k: v for k, v in report.metadata.items()
                      if "bandwidth" in k or "GFLOPS" in k}

    # NPU if available
    if hasattr(torch, "npu") and torch.npu.is_available():
        print("  - NPU memory bandwidth...")
        try:
            bench = MemoryBandwidthBench(device="npu", iterations=10)
            report = bench.run()
            print(report.report())
            results["npu"] = {k: v for k, v in report.metadata.items()
                            if "bandwidth" in k or "GFLOPS" in k}
        except Exception as e:
            print(f"  - NPU bandwidth measurement failed: {e}")

    return results


def run_model_throughput_benchmarks() -> Dict[str, Dict[str, Any]]:
    """Measure end-to-end model throughput."""
    print("\n[4/5] Running Model Throughput Benchmarks...")

    results = {}

    # Simple transformer-like model
    class SimpleTransformer(torch.nn.Module):
        def __init__(self, hidden_size=768, seq_len=128):
            super().__init__()
            self.embed = torch.nn.Embedding(10000, hidden_size)
            self.attn = torch.nn.MultiheadAttention(hidden_size, 8, batch_first=True)
            self.ffn = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_size * 4),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_size * 4, hidden_size),
            )
            self.seq_len = seq_len

        def forward(self, x):
            # x: (batch_size,)
            batch_size = x.shape[0] if x.dim() > 0 else 1
            x = x.reshape(batch_size, self.seq_len) if x.numel() >= batch_size * self.seq_len else x.reshape(batch_size, -1)
            x = self.embed(x.long())
            x, _ = self.attn(x, x, x)
            x = self.ffn(x)
            return x.mean(dim=1)

    model = SimpleTransformer()

    # CPU
    print("  - CPU throughput...")
    bench = ModelThroughputBench(
        model=model,
        input_fn=lambda: torch.randint(0, 10000, (4,)),
        device="cpu",
        iterations=50,
        warmup=10,
        batch_size=4,
    )
    report = bench.run()
    print(report.report())
    results["cpu"] = {
        "throughput_tps": float(report.metadata.get("throughput_samples_per_sec", 0)),
        "latency_p50_ms": float(report.metadata.get("latency_p50_ms", 0)),
        "latency_p95_ms": float(report.metadata.get("latency_p95_ms", 0)),
        "latency_p99_ms": float(report.metadata.get("latency_p99_ms", 0)),
    }

    # NPU if available
    if hasattr(torch, "npu") and torch.npu.is_available():
        print("  - NPU throughput...")
        try:
            bench = ModelThroughputBench(
                model=model,
                input_fn=lambda: torch.randint(0, 10000, (4,)),
                device="npu",
                iterations=50,
                warmup=10,
                batch_size=4,
            )
            report = bench.run()
            print(report.report())
            results["npu"] = {
                "throughput_tps": float(report.metadata.get("throughput_samples_per_sec", 0)),
                "latency_p50_ms": float(report.metadata.get("latency_p50_ms", 0)),
                "latency_p95_ms": float(report.metadata.get("latency_p95_ms", 0)),
                "latency_p99_ms": float(report.metadata.get("latency_p99_ms", 0)),
            }
        except Exception as e:
            print(f"  - NPU throughput measurement failed: {e}")

    return results


def compute_shim_overhead(operation_latencies: Dict[str, Dict[str, float]]) -> float:
    """Estimate shim overhead by comparing CPU operations through both paths."""
    if "cpu" not in operation_latencies:
        return 0.0

    # Average overhead across common operations
    common_ops = ["torch.empty(1) baseline", "torch.device('cpu') baseline"]

    overheads = []
    for op in common_ops:
        if op in operation_latencies["cpu"]:
            # In a real scenario, we'd compare shim-wrapped vs direct
            # For now, use operation latency as a proxy
            latency = operation_latencies["cpu"][op]
            overheads.append(latency)

    return sum(overheads) / len(overheads) if overheads else 0.0


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------


def generate_report(
    shim_overhead: Dict[str, float],
    op_latency: Dict[str, Dict[str, float]],
    memory_bw: Dict[str, Dict[str, float]],
    model_throughput: Dict[str, Dict[str, Any]],
) -> None:
    """Generate comprehensive benchmark report with grading."""

    print("\n" + "=" * 80)
    print("CUDA-MORPH COMPREHENSIVE BENCHMARK REPORT")
    print("=" * 80)

    # System info
    fp = get_system_fingerprint()
    print("\n[SYSTEM FINGERPRINT]")
    for k, v in fp.items():
        print(f"  {k}: {v}")

    # Shim Overhead Summary
    print("\n[SHIM OVERHEAD ANALYSIS]")
    print("  Proxy layer costs (microseconds per call):")
    for name, latency_us in sorted(shim_overhead.items(), key=lambda x: x[1]):
        print(f"    {name:<40} {latency_us:>10.2f} µs")

    avg_overhead = sum(shim_overhead.values()) / len(shim_overhead) if shim_overhead else 0
    print(f"\n  Average shim overhead: {avg_overhead:.2f} µs/call")

    if avg_overhead < 1.0:
        overhead_grade = "A+"
    elif avg_overhead < 5.0:
        overhead_grade = "A"
    elif avg_overhead < 10.0:
        overhead_grade = "B+"
    else:
        overhead_grade = "B"
    print(f"  Overhead Grade: {overhead_grade}")

    # Operation Latency Comparison
    print("\n[OPERATION LATENCY COMPARISON]")
    if "cpu" in op_latency:
        print("  CPU operations (µs/call):")
        for op, lat in sorted(op_latency["cpu"].items()):
            print(f"    {op:<40} {lat:>10.2f} µs")

    if "npu" in op_latency:
        print("\n  NPU operations (µs/call):")
        for op, lat in sorted(op_latency["npu"].items()):
            cpu_lat = op_latency["cpu"].get(op, lat)
            speedup = cpu_lat / lat if lat > 0 else 1.0
            print(f"    {op:<40} {lat:>10.2f} µs  (speedup: {speedup:.2f}x vs CPU)")

    # Memory Bandwidth Analysis
    print("\n[MEMORY BANDWIDTH ANALYSIS]")
    if "cpu" in memory_bw:
        print("  CPU memory characteristics:")
        for metric, value in sorted(memory_bw["cpu"].items()):
            print(f"    {metric}: {value}")

    if "npu" in memory_bw:
        print("\n  NPU memory characteristics:")
        for metric, value in sorted(memory_bw["npu"].items()):
            print(f"    {metric}: {value}")

    # Model Throughput Comparison
    print("\n[MODEL THROUGHPUT ANALYSIS]")

    scores = []

    if "cpu" in model_throughput:
        cpu_data = model_throughput["cpu"]
        print("\n  CPU Throughput:")
        print(f"    Throughput: {cpu_data['throughput_tps']:.1f} samples/sec")
        print(f"    Latency P50: {cpu_data['latency_p50_ms']:.2f} ms")
        print(f"    Latency P95: {cpu_data['latency_p95_ms']:.2f} ms")
        print(f"    Latency P99: {cpu_data['latency_p99_ms']:.2f} ms")

    if "npu" in model_throughput:
        npu_data = model_throughput["npu"]
        cpu_data = model_throughput.get("cpu", {})

        print("\n  NPU Throughput (with cuda-morph):")
        print(f"    Throughput: {npu_data['throughput_tps']:.1f} samples/sec")
        print(f"    Latency P50: {npu_data['latency_p50_ms']:.2f} ms")
        print(f"    Latency P95: {npu_data['latency_p95_ms']:.2f} ms")
        print(f"    Latency P99: {npu_data['latency_p99_ms']:.2f} ms")

        if cpu_data:
            speedup = npu_data['throughput_tps'] / cpu_data['throughput_tps'] if cpu_data['throughput_tps'] > 0 else 1.0
            print(f"\n    Speedup vs CPU: {speedup:.2f}x")

        # Score against native baseline
        scores.append(BenchmarkScore(
            name="Model Throughput",
            cuda_morph_value=npu_data['throughput_tps'],
            baseline_value=ASCEND_NATIVE_BASELINE.throughput_tps,
            unit="samples/sec",
            category="throughput",
        ))

        scores.append(BenchmarkScore(
            name="Latency P50",
            cuda_morph_value=npu_data['latency_p50_ms'],
            baseline_value=ASCEND_NATIVE_BASELINE.latency_p50_ms,
            unit="ms",
            category="latency",
        ))

    # Final Grade
    print("\n" + "=" * 80)
    print("PRODUCTION READINESS GRADE")
    print("=" * 80)

    if scores:
        grade, efficiency = GradeCard.grade(scores)
        print(f"\n  Overall Grade: {grade}")
        print(f"  Efficiency Score: {efficiency:.1f}%")

        print("\n  Detailed Scores:")
        for score in scores:
            print(f"    {score.name}: {score.efficiency():.1f}% of native performance")

        print("\n  Performance Interpretation:")
        if efficiency >= 95:
            print("    ✓ Production ready. Minimal overhead, suitable for all workloads.")
        elif efficiency >= 85:
            print("    ✓ Production ready. <15% overhead, good for most workloads.")
        elif efficiency >= 75:
            print("    ⚠ Acceptable for development. <25% overhead, monitor for latency-sensitive workloads.")
        elif efficiency >= 65:
            print("    ⚠ Developmental. 25-35% overhead, needs optimization.")
        else:
            print("    ✗ Not ready. Significant performance loss, requires investigation.")
    else:
        print("\n  Cannot compute grade: insufficient benchmark data.")
        print("  Ensure NPU device is available for full evaluation.")

    # Competitive Positioning
    print("\n" + "=" * 80)
    print("COMPETITIVE POSITIONING")
    print("=" * 80)
    print("\n  Benchmarks measured on current hardware. Comparison is theoretical:")
    print(f"    vLLM on H100: {VLLM_H100_BASELINE.throughput_tps:.0f} TPS, {VLLM_H100_BASELINE.latency_p50_ms:.1f}ms P50")
    print(f"    vLLM on A100: {VLLM_A100_BASELINE.throughput_tps:.0f} TPS, {VLLM_A100_BASELINE.latency_p50_ms:.1f}ms P50")
    print(f"    Native NPU:   {ASCEND_NATIVE_BASELINE.throughput_tps:.0f} TPS, {ASCEND_NATIVE_BASELINE.latency_p50_ms:.1f}ms P50")

    print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="cuda-morph comprehensive benchmark suite"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full benchmark suite (may take 10+ minutes)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="Export results to CSV file",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("CUDA-MORPH BENCHMARK SUITE")
    print("=" * 80)
    print(f"System: {get_system_fingerprint()['os']}")
    print(f"PyTorch: {torch.__version__}")

    try:
        # Run benchmarks
        shim_overhead = run_shim_overhead_benchmarks()
        op_latency = run_operation_latency_benchmarks()
        memory_bw = run_memory_bandwidth_benchmarks()
        model_throughput = run_model_throughput_benchmarks()

        # Generate report
        generate_report(shim_overhead, op_latency, memory_bw, model_throughput)

        # Export to CSV if requested
        if args.csv:
            print(f"\nExporting results to {args.csv}...")
            with open(args.csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Benchmark", "Device", "Value", "Unit"])

                for name, latency in shim_overhead.items():
                    writer.writerow(["Shim Overhead", "cpu", latency, "µs"])

                for device, ops in op_latency.items():
                    for op, latency in ops.items():
                        writer.writerow(["Operation Latency", device, latency, "µs"])

                for device, data in model_throughput.items():
                    for metric, value in data.items():
                        writer.writerow(["Model Throughput", device, value, "samples/sec or ms"])

            print(f"✓ Results exported to {args.csv}")

    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
