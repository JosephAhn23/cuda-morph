"""``cuda-morph scaffold`` — generate Ascend C operator project."""

from __future__ import annotations

from typing import Optional

import click


@click.command()
@click.argument("name")
@click.option("--pattern", default="elementwise",
              type=click.Choice(["elementwise", "reduction", "matmul", "custom"]),
              help="Computation pattern")
@click.option("--output", "-o", default=None, help="Output directory")
def scaffold(name: str, pattern: str, output: Optional[str]) -> None:
    """Generate Ascend C operator project from a template."""
    from ascend_compat.kernel_helper import OpSpec, scaffold as do_scaffold

    output_dir = output or f"./{name.lower()}_op"
    spec = OpSpec(
        name=name,
        inputs=[("x", "float16")],
        outputs=[("y", "float16")],
        pattern=pattern,
        description=f"Auto-generated {pattern} operator for Ascend NPU",
    )
    files = do_scaffold(spec, output_dir)
    click.echo(f"Scaffolded Ascend C operator '{name}' -> {output_dir}")
    for relpath in sorted(files.keys()):
        click.echo(f"  {relpath}")
