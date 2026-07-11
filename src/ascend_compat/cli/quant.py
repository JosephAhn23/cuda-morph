"""``cuda-morph quant`` — check quantization compatibility."""

from __future__ import annotations

import click


@click.command()
@click.argument("model")
def quant(model: str) -> None:
    """Check quantization compatibility for a model."""
    from ascend_compat.cuda_shim.quantization import check_model_quant, format_quant_report
    compat = check_model_quant(model)
    click.echo(format_quant_report(compat))
