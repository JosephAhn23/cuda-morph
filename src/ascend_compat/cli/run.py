"""``cuda-morph run`` — launch a script with full shims active."""

from __future__ import annotations

import os
import sys
from typing import Tuple

import click


@click.command()
@click.argument("script", type=click.Path(exists=True))
@click.argument("script_args", nargs=-1, type=click.UNPROCESSED)
def run(script: str, script_args: Tuple[str, ...]) -> None:
    """Run a script with full ascend-compat shims active."""
    import runpy

    os.environ["ASCEND_COMPAT_AUTO_ACTIVATE"] = "1"

    from ascend_compat.cuda_shim import activate
    activate()

    from ascend_compat.ecosystem._flash_attn_hook import install_flash_attn_hook
    install_flash_attn_hook()

    from ascend_compat.ecosystem import transformers_patch, deepspeed_patch, vllm_patch
    transformers_patch.apply()
    deepspeed_patch.apply()
    vllm_patch.apply()

    sys.argv = [script] + list(script_args)

    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as e:
        sys.exit(e.code if isinstance(e.code, int) else 0)
    except Exception as e:
        click.echo(f"Error running {script}: {e}", err=True)
        sys.exit(1)
