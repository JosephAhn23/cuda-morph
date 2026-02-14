"""``cuda-morph compile`` — show torch.compile backend info."""

from __future__ import annotations

import click


@click.command("compile")
def compile_cmd() -> None:
    """Show torch.compile backend info for Ascend."""
    from ascend_compat.cuda_shim.compile_helpers import get_compile_info, CompatibilityPolicy

    info = get_compile_info()
    click.echo("torch.compile configuration for Ascend:")
    click.echo(f"  Recommended backend:  {info['recommended_backend']}")
    click.echo(f"  torchair available:   {info['torchair_available']}")
    backends = info.get("available_backends", [])
    click.echo(f"  Registered backends:  {', '.join(backends) if backends else '(unknown)'}")

    try:
        is_tested = CompatibilityPolicy.check_forward_compat(policy="silent")
        from ascend_compat.cuda_shim.compile_helpers import LATEST_TESTED_VERSION
        tested_str = ".".join(str(v) for v in LATEST_TESTED_VERSION)
        click.echo(f"  Latest tested PyTorch: {tested_str}")
        click.echo(f"  Version in range:     {'yes' if is_tested else 'no (untested version)'}")
    except Exception:
        pass
