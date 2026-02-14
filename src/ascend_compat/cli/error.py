"""``cuda-morph error`` — translate CANN error codes."""

from __future__ import annotations

import click


@click.command()
@click.argument("code")
def error(code: str) -> None:
    """Translate a CANN error code to human-readable diagnosis."""
    from ascend_compat.doctor.error_codes import format_error
    click.echo(format_error(code))
