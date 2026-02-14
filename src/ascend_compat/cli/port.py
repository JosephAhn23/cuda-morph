"""``cuda-morph port`` — auto-add shim activation to CUDA files."""

from __future__ import annotations

import click


@click.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--dry-run", is_flag=True, help="Show changes without writing")
def port(file: str, dry_run: bool) -> None:
    """Auto-rewrite simple CUDA calls to ascend-compat."""
    from ascend_compat.cli._porter import port_file

    result = port_file(file, dry_run=dry_run)
    if dry_run:
        click.echo("--- Dry run output ---")
        click.echo(result)
