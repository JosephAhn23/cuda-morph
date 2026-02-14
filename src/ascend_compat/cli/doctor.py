"""``cuda-morph doctor`` — environment diagnostics."""

from __future__ import annotations

import click


@click.command()
@click.option("--full", is_flag=True, help="Run deep environment validation")
def doctor(full: bool) -> None:
    """Run environment diagnostics (versions, hardware, compatibility)."""
    if full:
        from ascend_compat.doctor.env_setup import full_environment_check, format_env_report
        results = full_environment_check()
        click.echo(format_env_report(results))
    else:
        from ascend_compat.doctor.version_check import check_versions, format_report
        results = check_versions()
        click.echo(format_report(results))
