"""``cuda-morph bench`` — run benchmarks."""

from __future__ import annotations

from typing import Optional

import click


@click.command()
@click.argument("mode", default="overhead", type=click.Choice(["overhead", "ops", "bandwidth"]))
@click.option("--device", default="cpu", help="Device for ops benchmark")
@click.option("--iterations", type=int, default=None, help="Number of iterations")
@click.option("--csv", "csv_file", default=None, help="Export results to CSV file")
def bench(mode: str, device: str, iterations: Optional[int], csv_file: Optional[str]) -> None:
    """Run benchmarks (overhead measurement, op latency, memory bandwidth)."""
    from ascend_compat.bench import ShimOverheadBench, OpLatencyBench, MemoryBandwidthBench

    if mode == "overhead":
        iters = iterations or 50000
        report = ShimOverheadBench(iterations=iters).run()
    elif mode == "bandwidth":
        iters = iterations or 50
        report = MemoryBandwidthBench(device=device, iterations=iters).run()
    else:
        iters = iterations or 1000
        report = OpLatencyBench(device=device, iterations=iters).run()

    click.echo(report.report())
    if csv_file:
        with open(csv_file, "w", newline="") as f:
            f.write(report.to_csv())
        click.echo(f"\nResults exported to {csv_file}")
