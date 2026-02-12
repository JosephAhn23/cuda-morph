"""Ascend C operator development scaffolding.

Writing a custom Ascend C operator requires coordinating 5-7 files across
host-side and device-side code — significantly more boilerplate than CUDA's
single .cu file.  This package provides:

1. ``scaffold()`` — Generate a complete operator project from a template
2. ``OpSpec`` — Declarative operator specification
3. Templates for common patterns (elementwise, reduction, matmul)

The generated code follows Huawei's recommended 3-stage pipeline pattern:
    CopyIn → Compute → CopyOut
with double-buffering for DMA latency hiding.

Usage::

    from ascend_compat.kernel_helper import scaffold, OpSpec

    spec = OpSpec(
        name="MyCustomAdd",
        inputs=[("x", "float16"), ("y", "float16")],
        outputs=[("z", "float16")],
        pattern="elementwise",
    )
    scaffold(spec, output_dir="./my_custom_add")
"""

from ascend_compat.kernel_helper.spec import OpSpec
from ascend_compat.kernel_helper.scaffold import scaffold

__all__ = ["OpSpec", "scaffold"]
