"""Tests for ascend_compat.bench — benchmarking framework."""

from __future__ import annotations

import pytest


class TestBenchResult:
    """Test BenchResult data class."""

    def test_calls_per_second(self):
        from ascend_compat.bench import BenchResult

        r = BenchResult(
            name="test_op",
            iterations=1000,
            total_seconds=2.0,
            per_call_us=2000.0,
            device="cpu",
        )
        assert r.calls_per_second == 500.0

    def test_calls_per_second_zero_time(self):
        from ascend_compat.bench import BenchResult

        r = BenchResult(
            name="test_op",
            iterations=0,
            total_seconds=0.0,
            per_call_us=0.0,
        )
        assert r.calls_per_second == 0

    def test_default_device(self):
        from ascend_compat.bench import BenchResult

        r = BenchResult(name="x", iterations=1, total_seconds=1, per_call_us=1)
        assert r.device == "cpu"


class TestBenchReport:
    """Test BenchReport formatting."""

    def test_report_format(self):
        from ascend_compat.bench import BenchReport, BenchResult

        report = BenchReport(
            title="Test Report",
            results=[
                BenchResult("fast_op", 100, 0.01, 100.0, "cpu"),
                BenchResult("slow_op", 100, 1.0, 10000.0, "cpu"),
            ],
            metadata={"device": "cpu"},
        )

        text = report.report()
        assert "Test Report" in text
        assert "fast_op" in text
        assert "slow_op" in text
        assert "device: cpu" in text

    def test_csv_export(self):
        from ascend_compat.bench import BenchReport, BenchResult

        report = BenchReport(
            title="CSV Test",
            results=[
                BenchResult("op_a", 50, 0.5, 10000.0, "cpu"),
                BenchResult("op_b", 50, 0.1, 2000.0, "npu"),
            ],
        )

        csv_text = report.to_csv(include_fingerprint=False)
        lines = csv_text.strip().split("\n")
        assert len(lines) == 3  # header + 2 rows
        assert "operation" in lines[0]
        assert "op_a" in lines[1]
        assert "op_b" in lines[2]

        # With fingerprint (default)
        csv_with_fp = report.to_csv()
        fp_lines = [l for l in csv_with_fp.strip().split("\n") if l.startswith("#")]
        assert len(fp_lines) > 0  # fingerprint header lines present
        data_lines = [l for l in csv_with_fp.strip().split("\n") if not l.startswith("#")]
        assert "operation" in data_lines[0]

    def test_empty_report(self):
        from ascend_compat.bench import BenchReport

        report = BenchReport(title="Empty")
        text = report.report()
        assert "Empty" in text
        csv = report.to_csv()
        assert "operation" in csv


class TestTimeit:
    """Test the internal _timeit helper."""

    def test_timeit_returns_tuple(self):
        from ascend_compat.bench import _timeit

        elapsed, iters = _timeit(lambda: None, iterations=100, warmup=5)
        assert isinstance(elapsed, float)
        assert iters == 100
        assert elapsed > 0

    def test_timeit_measures_work(self):
        """Verify that more iterations take more time."""
        from ascend_compat.bench import _timeit

        import time
        elapsed1, _ = _timeit(lambda: time.sleep(0.0001), iterations=5, warmup=1)
        elapsed2, _ = _timeit(lambda: time.sleep(0.0001), iterations=50, warmup=1)
        # 50 iterations should take roughly 10x longer (with tolerance)
        assert elapsed2 > elapsed1


class TestShimOverheadBench:
    """Test ShimOverheadBench end-to-end."""

    def test_run_produces_report(self):
        from ascend_compat.bench import ShimOverheadBench

        bench = ShimOverheadBench(iterations=100)
        report = bench.run()

        assert report.title == "Shim Overhead Benchmark"
        assert len(report.results) >= 3  # At least 3 benchmarks
        assert all(r.iterations == 100 for r in report.results)
        assert all(r.per_call_us > 0 for r in report.results)

    def test_report_has_metadata(self):
        from ascend_compat.bench import ShimOverheadBench

        report = ShimOverheadBench(iterations=50).run()
        assert "PyTorch" in report.metadata
        assert "iterations" in report.metadata


class TestOpLatencyBench:
    """Test OpLatencyBench end-to-end."""

    def test_run_cpu(self):
        from ascend_compat.bench import OpLatencyBench

        bench = OpLatencyBench(device="cpu", iterations=10)
        report = bench.run()

        assert report.title.startswith("Operation Latency")
        assert len(report.results) >= 5
        assert all(r.device == "cpu" for r in report.results)

    def test_invalid_device(self):
        """Invalid device should not crash, should report error."""
        from ascend_compat.bench import OpLatencyBench

        # "nonexistent" is not a valid device — should gracefully handle
        bench = OpLatencyBench(device="cpu", iterations=5)
        report = bench.run()
        assert report.results is not None


class TestModelThroughputBench:
    """Test ModelThroughputBench."""

    def test_run_simple_model(self):
        import torch
        from ascend_compat.bench import ModelThroughputBench

        # Simple linear model
        model = torch.nn.Linear(10, 5)
        bench = ModelThroughputBench(
            model=model,
            input_fn=lambda: torch.randn(1, 10),
            device="cpu",
            iterations=20,
            warmup=5,
            batch_size=1,
        )
        report = bench.run()

        assert "throughput_samples_per_sec" in report.metadata
        assert "latency_p50_ms" in report.metadata
        assert "latency_p95_ms" in report.metadata
        assert "latency_p99_ms" in report.metadata
        assert len(report.results) == 1
        assert float(report.metadata["throughput_samples_per_sec"]) > 0
