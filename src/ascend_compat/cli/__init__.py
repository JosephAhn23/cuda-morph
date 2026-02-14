"""CLI package for cuda-morph.

Subcommands are registered from separate modules for maintainability.
Each subcommand file is self-contained and independently testable.

Usage::

    cuda-morph check model.py
    cuda-morph port model.py
    cuda-morph doctor
    cuda-morph info
    cuda-morph verify --device npu
    cuda-morph bench overhead
"""

from __future__ import annotations

import click

from ascend_compat._logging import get_logger

logger = get_logger(__name__)


@click.group()
@click.version_option(package_name="ascend-compat")
def main() -> None:
    """cuda-morph: Run CUDA code on any accelerator."""


# Register subcommands from separate modules
from ascend_compat.cli.check import check  # noqa: E402
from ascend_compat.cli.port import port  # noqa: E402
from ascend_compat.cli.doctor import doctor  # noqa: E402
from ascend_compat.cli.info import info  # noqa: E402
from ascend_compat.cli.verify import verify  # noqa: E402
from ascend_compat.cli.bench import bench  # noqa: E402
from ascend_compat.cli.run import run  # noqa: E402
from ascend_compat.cli.security import security  # noqa: E402
from ascend_compat.cli.scaffold import scaffold  # noqa: E402
from ascend_compat.cli.compile import compile_cmd  # noqa: E402
from ascend_compat.cli.error import error  # noqa: E402
from ascend_compat.cli.quant import quant  # noqa: E402
from ascend_compat.cli.vllm import vllm  # noqa: E402

main.add_command(check)
main.add_command(port)
main.add_command(doctor)
main.add_command(info)
main.add_command(verify)
main.add_command(bench)
main.add_command(run)
main.add_command(security)
main.add_command(scaffold)
main.add_command(compile_cmd, name="compile")
main.add_command(error)
main.add_command(quant)
main.add_command(vllm)


# ---------------------------------------------------------------------------
# Backward compatibility re-exports (old monolithic cli.py imports)
# ---------------------------------------------------------------------------
from ascend_compat.cli._scanner import CheckReport, CudaDependency, check_file  # noqa: F401, E402
from ascend_compat.cli._porter import port_file  # noqa: F401, E402
from ascend_compat.cli._info import show_info  # noqa: F401, E402
