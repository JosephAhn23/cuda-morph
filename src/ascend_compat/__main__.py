"""Support ``python -m ascend_compat`` as a launcher.

Usage::

    # Run a script with all shims active:
    python -m ascend_compat run train.py --batch-size 32

    # Equivalent to adding ``import ascend_compat`` at the top of train.py,
    # plus auto-applying ecosystem patches.

    # Or use any CLI command:
    python -m ascend_compat doctor
    python -m ascend_compat check model.py
    python -m ascend_compat error 507035

The ``run`` subcommand is the primary addition here.  It:
1. Activates the cuda_shim (torch.cuda → torch.npu)
2. Installs the flash_attn import hook
3. Applies ecosystem patches (transformers, deepspeed)
4. Executes the user's script with full compatibility
"""

from __future__ import annotations

import sys


def main() -> None:
    from ascend_compat.cli import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()
