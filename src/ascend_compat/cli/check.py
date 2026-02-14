"""``cuda-morph check`` — scan a Python file for CUDA dependencies."""

from __future__ import annotations

import click


@click.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def check(file: str, as_json: bool) -> None:
    """Scan a Python file for CUDA dependencies."""
    from ascend_compat.cli._scanner import check_file

    report = check_file(file)
    if as_json:
        import json

        data = {
            "file": report.file_path,
            "total_cuda_refs": report.total_cuda_refs,
            "migration_difficulty": report.migration_difficulty,
            "dependencies": [
                {
                    "api_call": d.api_call,
                    "line": d.line_number,
                    "status": d.status,
                    "suggestion": d.suggestion,
                }
                for d in report.dependencies
            ],
        }
        click.echo(json.dumps(data, indent=2))
    else:
        click.echo(report.summary())
