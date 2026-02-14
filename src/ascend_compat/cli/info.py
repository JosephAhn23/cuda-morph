"""``cuda-morph info`` — show system info and shim status."""

from __future__ import annotations

import click


@click.command()
def info() -> None:
    """Show system info and shim status."""
    from ascend_compat.cli._info import show_info
    click.echo(show_info())
