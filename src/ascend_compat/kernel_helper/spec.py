"""Declarative operator specification for Ascend C scaffolding.

An OpSpec describes what a custom operator does at a high level —
its inputs, outputs, data types, and computation pattern — without
requiring the developer to understand Ascend C's tiling, memory
management, or pipeline orchestration.  The scaffold tool uses
the spec to generate all required files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple


# Ascend C supports these data types in the Cube and Vector units
SUPPORTED_DTYPES = frozenset({
    "float16", "fp16",
    "float32", "fp32",
    "bfloat16", "bf16",
    "int8",
    "int32",
    "int16",
    "uint8",
})

# Patterns map to different code generation templates
SUPPORTED_PATTERNS = frozenset({
    "elementwise",   # y = f(x)  — per-element, Vector unit
    "reduction",     # y = reduce(x, dim)  — Vector unit
    "matmul",        # C = A @ B  — Cube unit (16×16×16 MAC)
    "custom",        # User provides their own compute body
})


@dataclass
class OpSpec:
    """Declarative specification for an Ascend C custom operator.

    Attributes:
        name: Operator name (PascalCase, e.g. ``FusedRMSNorm``).
        inputs: List of (name, dtype) tuples for input tensors.
        outputs: List of (name, dtype) tuples for output tensors.
        pattern: Computation pattern — one of ``elementwise``,
                 ``reduction``, ``matmul``, ``custom``.
        attrs: Extra scalar attributes (name → C++ type) passed to kernel.
        workspace_bytes: Extra workspace memory needed per AI Core.
        description: Human-readable description for generated comments.
        alignment: Data alignment requirement in bytes (default 32).
        supports_bf16: Whether to generate BF16 variants. BF16 support
                       varies by hardware generation (910B+ only).
    """

    name: str
    inputs: List[Tuple[str, str]]
    outputs: List[Tuple[str, str]]
    pattern: str = "elementwise"
    attrs: Dict[str, str] = field(default_factory=dict)
    workspace_bytes: int = 0
    description: str = ""
    alignment: int = 32
    supports_bf16: bool = True

    def __post_init__(self) -> None:
        """Validate the specification."""
        if not self.name:
            raise ValueError("OpSpec.name cannot be empty")

        if not self.name[0].isupper():
            raise ValueError(
                f"OpSpec.name should be PascalCase, got '{self.name}'. "
                "Example: 'FusedRMSNorm', 'MyCustomAdd'"
            )

        if self.pattern not in SUPPORTED_PATTERNS:
            raise ValueError(
                f"Unknown pattern '{self.pattern}'. "
                f"Supported: {sorted(SUPPORTED_PATTERNS)}"
            )

        for label, tensors in [("inputs", self.inputs), ("outputs", self.outputs)]:
            if not tensors:
                raise ValueError(f"OpSpec must have at least one {label}")
            for tname, dtype in tensors:
                norm_dtype = dtype.lower().replace("float", "fp").replace("bfloat", "bf")
                # Normalize common aliases
                canonical = dtype.lower()
                if canonical not in SUPPORTED_DTYPES:
                    raise ValueError(
                        f"{label} tensor '{tname}' has unsupported dtype "
                        f"'{dtype}'. Supported: {sorted(SUPPORTED_DTYPES)}"
                    )

        if self.alignment % 32 != 0:
            raise ValueError(
                f"Alignment must be a multiple of 32 bytes (Ascend requirement), "
                f"got {self.alignment}"
            )

    @property
    def all_tensors(self) -> List[Tuple[str, str]]:
        """All input and output tensors."""
        return list(self.inputs) + list(self.outputs)

    @property
    def uses_cube(self) -> bool:
        """Whether this operator uses the Cube unit (matrix multiply)."""
        return self.pattern == "matmul"

    @property
    def uses_vector(self) -> bool:
        """Whether this operator uses the Vector unit."""
        return self.pattern in ("elementwise", "reduction", "custom")

    def to_dict(self) -> Dict:
        """Serialize to a dict (useful for templates)."""
        return {
            "name": self.name,
            "inputs": [{"name": n, "dtype": d} for n, d in self.inputs],
            "outputs": [{"name": n, "dtype": d} for n, d in self.outputs],
            "pattern": self.pattern,
            "attrs": self.attrs,
            "workspace_bytes": self.workspace_bytes,
            "description": self.description,
            "alignment": self.alignment,
            "uses_cube": self.uses_cube,
            "uses_vector": self.uses_vector,
        }
