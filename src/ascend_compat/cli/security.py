"""``cuda-morph security`` — verify binary integrity."""

from __future__ import annotations

import sys

import click


@click.command()
def security() -> None:
    """Verify torch_npu and CANN binary integrity.

    NOTE: The hash database is currently empty. Results only check file
    presence, not actual integrity.
    """
    click.echo(
        "NOTE: Security verification is incomplete.\n"
        "The hash database (_KNOWN_HASHES) is empty — no torch_npu releases\n"
        "have been fingerprinted yet. Results below only check file presence,\n"
        "not actual integrity. See doctor/security_check.py to contribute hashes.\n"
    )
    from ascend_compat.doctor.security_check import full_security_check, format_security_report

    results = full_security_check()
    click.echo(format_security_report(results))
    errors = sum(1 for r in results if r.status == "error")
    if errors:
        sys.exit(1)
