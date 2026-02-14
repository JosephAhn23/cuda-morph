"""CUDA dependency scanner (AST + regex). Used by cuda-morph check."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CudaDependency:
    """A single CUDA dependency found in the source code."""

    api_call: str           # e.g. "torch.cuda.is_available"
    line_number: int        # 1-based line number
    line_text: str          # The source line (stripped)
    status: str             # "transparent", "needs_wrapper", "unsupported", "unknown"
    suggestion: Optional[str] = None  # Migration suggestion


@dataclass
class CheckReport:
    """Report from scanning a file for CUDA dependencies."""

    file_path: str
    total_cuda_refs: int = 0
    dependencies: List[CudaDependency] = field(default_factory=list)
    imports_torch_cuda: bool = False
    imports_cudnn: bool = False
    has_cuda_device_strings: bool = False
    has_dot_cuda_calls: bool = False  # tensor.cuda() / model.cuda()
    migration_difficulty: str = "trivial"  # trivial / easy / moderate / hard

    def summary(self) -> str:
        """Return a human-readable summary of the report."""
        lines = [
            f"╔══════════════════════════════════════════════════════════════╗",
            f"║  ascend-compat migration check: {Path(self.file_path).name:<28}║",
            f"╠══════════════════════════════════════════════════════════════╣",
            f"║  Total CUDA references:  {self.total_cuda_refs:<34}║",
            f"║  Migration difficulty:   {self.migration_difficulty:<34}║",
            f"╠══════════════════════════════════════════════════════════════╣",
        ]

        # Group by status
        by_status: Dict[str, List[CudaDependency]] = {}
        for dep in self.dependencies:
            by_status.setdefault(dep.status, []).append(dep)

        status_labels = {
            "transparent": "✅ Transparent (work as-is)",
            "needs_wrapper": "🔄 Needs wrapper (ascend-compat handles)",
            "unsupported": "❌ Unsupported (manual rewrite needed)",
            "unknown": "❓ Unknown (test on Ascend hardware)",
        }

        for status, label in status_labels.items():
            deps = by_status.get(status, [])
            if deps:
                lines.append(f"║                                                              ║")
                lines.append(f"║  {label:<58}║")
                for dep in deps:
                    loc = f"  L{dep.line_number}: {dep.api_call}"
                    lines.append(f"║  {loc:<58}║")
                    if dep.suggestion:
                        sug = f"    → {dep.suggestion}"
                        # Truncate long suggestions
                        if len(sug) > 58:
                            sug = sug[:55] + "..."
                        lines.append(f"║  {sug:<58}║")

        # Quick fixes section
        if self.has_cuda_device_strings or self.has_dot_cuda_calls:
            lines.append(f"║                                                              ║")
            lines.append(f"║  Quick fixes:                                                ║")
            if self.has_cuda_device_strings:
                lines.append(f'║    • Replace "cuda" strings with ascend_compat.device...    ║')
            if self.has_dot_cuda_calls:
                lines.append(f"║    • Replace .cuda() with ascend_compat.device.to_device()  ║")
            lines.append(f"║    • Or: just add 'import ascend_compat' at the top         ║")

        lines.append(f"╚══════════════════════════════════════════════════════════════╝")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Known CUDA patterns and their compatibility status
# ---------------------------------------------------------------------------

# Map of API call pattern → (status, suggestion)
_CUDA_PATTERNS: Dict[str, Tuple[str, Optional[str]]] = {
    # Device management
    "torch.cuda.is_available": ("needs_wrapper", "Use ascend_compat.device.is_available()"),
    "torch.cuda.device_count": ("needs_wrapper", "Use ascend_compat.device.device_count()"),
    "torch.cuda.current_device": ("needs_wrapper", "Use ascend_compat.device.current_device()"),
    "torch.cuda.set_device": ("needs_wrapper", "Use ascend_compat.device.set_device()"),
    "torch.cuda.get_device_name": ("needs_wrapper", "Use ascend_compat.device.get_device_name()"),
    "torch.cuda.get_device_properties": (
        "needs_wrapper", "Use ascend_compat.device.get_device_properties()"
    ),
    # Memory
    "torch.cuda.memory_allocated": ("needs_wrapper", "Use ascend_compat.memory.memory_allocated()"),
    "torch.cuda.max_memory_allocated": (
        "needs_wrapper", "Use ascend_compat.memory.max_memory_allocated()"
    ),
    "torch.cuda.empty_cache": ("needs_wrapper", "Use ascend_compat.memory.empty_cache()"),
    "torch.cuda.memory_reserved": ("needs_wrapper", "Use ascend_compat.memory.memory_reserved()"),
    "torch.cuda.reset_peak_memory_stats": (
        "needs_wrapper", "Use ascend_compat.memory.reset_peak_memory_stats()"
    ),
    "torch.cuda.memory_summary": ("needs_wrapper", "Use ascend_compat.memory.memory_summary()"),
    "torch.cuda.memory_snapshot": ("unsupported", "No Ascend equivalent — remove or guard"),
    # Streams
    "torch.cuda.synchronize": ("needs_wrapper", "Use ascend_compat.streams.synchronize()"),
    "torch.cuda.Stream": ("needs_wrapper", "Use ascend_compat.streams.Stream()"),
    "torch.cuda.Event": ("needs_wrapper", "Use ascend_compat.streams.Event()"),
    "torch.cuda.current_stream": ("needs_wrapper", "Use ascend_compat.streams.current_stream()"),
    # AMP
    "torch.cuda.amp.autocast": ("needs_wrapper", "Use ascend_compat.ops.autocast()"),
    "torch.cuda.amp.GradScaler": ("needs_wrapper", "Use ascend_compat.ops.GradScaler()"),
    # Seeds
    "torch.cuda.manual_seed": ("needs_wrapper", "Use ascend_compat.ops.manual_seed()"),
    "torch.cuda.manual_seed_all": ("needs_wrapper", "Use ascend_compat.ops.manual_seed_all()"),
    # cuDNN
    "torch.backends.cudnn.benchmark": (
        "needs_wrapper", "Safe to keep; no-op on Ascend via CudnnShim"
    ),
    "torch.backends.cudnn.deterministic": (
        "needs_wrapper", "Maps to torch.use_deterministic_algorithms()"
    ),
    "torch.backends.cudnn.enabled": ("needs_wrapper", "Safe to keep; shim handles it"),
    # CUDA Graphs
    "torch.cuda.CUDAGraph": ("unsupported", "Use ascend_compat.ops.graph_mode() instead"),
    "torch.cuda.graph": ("unsupported", "Use ascend_compat.ops.graph_mode() instead"),
    # Profiling
    "torch.cuda.nvtx": ("unsupported", "Use Ascend msprof profiling tools"),
    # Distributed
    "nccl": ("needs_wrapper", "Use ascend_compat.ops.get_distributed_backend()"),
}


# ---------------------------------------------------------------------------
# AST-based scanner
# ---------------------------------------------------------------------------


class _CudaVisitor(ast.NodeVisitor):
    """AST visitor that finds CUDA-specific patterns in Python source."""

    def __init__(self, source_lines: List[str]) -> None:
        self.source_lines = source_lines
        self.found: List[CudaDependency] = []
        self.imports_torch_cuda = False
        self.imports_cudnn = False

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            if "torch_npu" in (alias.name or ""):
                pass  # Already using torch_npu
            if "cuda" in (alias.name or "").lower():
                self.imports_torch_cuda = True

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        module = node.module or ""
        if "torch.cuda" in module:
            self.imports_torch_cuda = True
        if "cudnn" in module:
            self.imports_cudnn = True
        for alias in node.names:
            name = alias.name or ""
            if "cuda" in name.lower():
                self.imports_torch_cuda = True

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        """Detect attribute chains like torch.cuda.is_available."""
        chain = self._get_attr_chain(node)
        if chain:
            self._check_pattern(chain, node.lineno)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        """Detect .cuda() calls on objects (tensor.cuda(), model.cuda())."""
        if isinstance(node.func, ast.Attribute) and node.func.attr == "cuda":
            line_text = self._get_line(node.lineno)
            self.found.append(CudaDependency(
                api_call=".cuda()",
                line_number=node.lineno,
                line_text=line_text,
                status="needs_wrapper",
                suggestion="Replace with ascend_compat.device.to_device(obj)",
            ))
        self.generic_visit(node)

    def _get_attr_chain(self, node: ast.AST) -> Optional[str]:
        """Reconstruct an attribute chain like 'torch.cuda.is_available'."""
        parts: List[str] = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        parts.reverse()
        return ".".join(parts) if parts else None

    def _check_pattern(self, chain: str, lineno: int) -> None:
        """Check if an attribute chain matches a known CUDA pattern."""
        for pattern, (status, suggestion) in _CUDA_PATTERNS.items():
            if chain.startswith(pattern) or chain == pattern:
                self.found.append(CudaDependency(
                    api_call=chain,
                    line_number=lineno,
                    line_text=self._get_line(lineno),
                    status=status,
                    suggestion=suggestion,
                ))
                return

        # Generic torch.cuda.* detection — but skip bare "torch.cuda" which
        # is just the namespace itself (not an API call).  Also skip short
        # partial matches that are sub-expressions of known patterns.
        if "torch.cuda" in chain and chain not in ("torch.cuda", "torch.backends"):
            # Check if this is a prefix of a known pattern (e.g. "torch.cuda.amp")
            is_namespace_only = any(
                p.startswith(chain + ".") for p in _CUDA_PATTERNS
            )
            if not is_namespace_only:
                self.found.append(CudaDependency(
                    api_call=chain,
                    line_number=lineno,
                    line_text=self._get_line(lineno),
                    status="unknown",
                    suggestion="Check Ascend compatibility for this API",
                ))

    def _get_line(self, lineno: int) -> str:
        """Get the source line (1-indexed)."""
        if 1 <= lineno <= len(self.source_lines):
            return self.source_lines[lineno - 1].strip()
        return ""


# ---------------------------------------------------------------------------
# Regex-based scanner (catches string literals the AST misses)
# ---------------------------------------------------------------------------

# Patterns for CUDA device strings in code
_CUDA_STRING_RE = re.compile(
    r"""(?:['"])cuda(?::?\d*)?(?:['"])""",
    re.IGNORECASE,
)

_DOT_CUDA_RE = re.compile(r"""\.cuda\s*\(""")


def _scan_strings(source: str, lines: List[str]) -> Tuple[bool, bool, List[CudaDependency]]:
    """Scan for CUDA device strings and .cuda() calls via regex.

    Returns:
        (has_cuda_strings, has_dot_cuda, additional_deps)
    """
    has_cuda_strings = False
    has_dot_cuda = False
    deps: List[CudaDependency] = []

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue  # Skip comments

        if _CUDA_STRING_RE.search(line):
            has_cuda_strings = True
            deps.append(CudaDependency(
                api_call='"cuda" device string',
                line_number=i,
                line_text=stripped,
                status="needs_wrapper",
                suggestion='Add "import ascend_compat" or use device.get_device_string()',
            ))

    return has_cuda_strings, has_dot_cuda, deps


# ---------------------------------------------------------------------------
# Main check function
# ---------------------------------------------------------------------------


def check_file(file_path: str) -> CheckReport:
    """Scan a Python file for CUDA dependencies.

    Args:
        file_path: Path to the Python file to scan.

    Returns:
        A :class:`CheckReport` with all findings.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()

    report = CheckReport(file_path=str(path))

    # AST pass
    try:
        tree = ast.parse(source, filename=str(path))
        visitor = _CudaVisitor(lines)
        visitor.visit(tree)
        report.dependencies.extend(visitor.found)
        report.imports_torch_cuda = visitor.imports_torch_cuda
        report.imports_cudnn = visitor.imports_cudnn
    except SyntaxError as e:
        report.dependencies.append(CudaDependency(
            api_call="<syntax error>",
            line_number=e.lineno or 0,
            line_text=str(e),
            status="unknown",
            suggestion="Fix syntax error before checking",
        ))

    # Regex pass
    has_strings, has_dot_cuda, string_deps = _scan_strings(source, lines)
    report.has_cuda_device_strings = has_strings
    report.has_dot_cuda_calls = has_dot_cuda
    # Only add string deps that aren't duplicates of AST findings
    ast_lines = {d.line_number for d in report.dependencies}
    for dep in string_deps:
        if dep.line_number not in ast_lines:
            report.dependencies.append(dep)

    # Calculate totals
    report.total_cuda_refs = len(report.dependencies)

    # Determine migration difficulty
    report.migration_difficulty = _assess_difficulty(report)

    # Sort by line number
    report.dependencies.sort(key=lambda d: d.line_number)

    return report


def _assess_difficulty(report: CheckReport) -> str:
    """Assess overall migration difficulty based on findings."""
    unsupported = sum(1 for d in report.dependencies if d.status == "unsupported")
    unknown = sum(1 for d in report.dependencies if d.status == "unknown")
    needs_wrapper = sum(1 for d in report.dependencies if d.status == "needs_wrapper")

    if report.total_cuda_refs == 0:
        return "trivial"
    elif unsupported > 0:
        return "hard"
    elif unknown > 3:
        return "moderate"
    elif needs_wrapper > 0 and unknown == 0:
        return "easy"
    else:
        return "moderate"
