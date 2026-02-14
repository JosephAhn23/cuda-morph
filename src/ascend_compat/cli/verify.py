"""``cuda-morph verify`` — empirically verify operator correctness."""

from __future__ import annotations

import sys

import click


@click.command()
@click.option("--device", default="cpu", help="Device to verify on: npu, cuda, or cpu")
def verify(device: str) -> None:
    """Empirically verify operator correctness on current device."""
    from ascend_compat.validation import OperatorVerifier

    verifier = OperatorVerifier(device=device)
    click.echo(f"Running operator verification on device={device}...")
    click.echo(f"(Use --device npu on Ascend hardware for real validation)\n")
    results = verifier.run_all()
    click.echo(verifier.format_report(results))
    failed = sum(1 for r in results if not r.passed)
    if failed:
        sys.exit(1)
