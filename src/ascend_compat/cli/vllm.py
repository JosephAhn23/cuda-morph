"""``cuda-morph vllm`` — check vLLM readiness."""

from __future__ import annotations

import click


@click.command()
def vllm() -> None:
    """Check vLLM/vllm-ascend readiness."""
    from ascend_compat.ecosystem.vllm_patch import check_vllm_readiness

    result = check_vllm_readiness()
    icon = "[OK]" if result["ready"] else "[XX]"
    click.echo(f"{icon} vLLM readiness: {'Ready' if result['ready'] else 'Not ready'}")
    for k, v in result["info"].items():
        click.echo(f"  {k}: {v}")
    for issue in result["issues"]:
        click.echo(f"  [!!] {issue}")
