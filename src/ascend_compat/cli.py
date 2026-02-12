"""CLI tool for CUDA → Ascend migration analysis and automated porting.

Provides two main commands:

``ascend-compat check <file.py>``
    Scan a Python file for CUDA dependencies and report migration difficulty.
    Produces a structured report showing which CUDA calls are transparent,
    which need wrappers, and which are unsupported on Ascend.

``ascend-compat port <file.py>``
    Auto-rewrite simple CUDA calls to ascend-compat equivalents.  Creates
    a backup of the original file and writes the ported version.

``ascend-compat info``
    Show detected hardware and current shim status.

Architecture
------------
The scanner uses a two-pass approach:
1. **AST pass** — parse the Python AST to find import statements and
   attribute access patterns (``torch.cuda.X``, ``torch.backends.cudnn.X``)
2. **Regex pass** — catch string literals like ``"cuda"`` and ``"cuda:0"``
   that the AST pass might miss

This is deliberately simple (no type inference, no flow analysis) because
the goal is to give developers a fast overview, not a perfect analysis.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


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


# ---------------------------------------------------------------------------
# Port function (auto-rewrite)
# ---------------------------------------------------------------------------


def port_file(file_path: str, dry_run: bool = False) -> str:
    """Auto-rewrite simple CUDA calls to ascend-compat equivalents.

    This handles the "easy" migrations:
    - Adds ``import ascend_compat`` at the top of the file
    - Replaces ``torch.cuda.is_available()`` with
      ``ascend_compat.device.is_available()`` (and similar)
    - Replaces ``"cuda"`` device strings with
      ``ascend_compat.device.get_device_string()``
    - Replaces ``.cuda()`` calls with
      ``ascend_compat.device.to_device(...)``

    Args:
        file_path: Path to the Python file to port.
        dry_run: If True, return the modified source without writing.

    Returns:
        The modified source code.
    """
    path = Path(file_path)
    source = path.read_text(encoding="utf-8")

    # Create backup
    if not dry_run:
        backup_path = path.with_suffix(path.suffix + ".bak")
        backup_path.write_text(source, encoding="utf-8")
        print(f"  Backup saved to: {backup_path}")

    modified = source

    # 1. Add import ascend_compat after torch imports
    if "import ascend_compat" not in modified:
        # Find the last torch import and add after it
        import_pattern = re.compile(r"^(import torch.*|from torch.* import.*)$", re.MULTILINE)
        matches = list(import_pattern.finditer(modified))
        if matches:
            last_match = matches[-1]
            insert_pos = last_match.end()
            modified = (
                modified[:insert_pos]
                + "\nimport ascend_compat  # CUDA→Ascend compatibility shim"
                + modified[insert_pos:]
            )
        else:
            # No torch imports found — add at the top
            modified = (
                "import ascend_compat  # CUDA→Ascend compatibility shim\n" + modified
            )

    # 2. Simple string replacements (conservative — only exact patterns)
    replacements = [
        # Device management
        ("torch.cuda.is_available()", "ascend_compat.device.is_available()"),
        ("torch.cuda.device_count()", "ascend_compat.device.device_count()"),
        ("torch.cuda.current_device()", "ascend_compat.device.current_device()"),
        ("torch.cuda.empty_cache()", "ascend_compat.memory.empty_cache()"),
        ("torch.cuda.synchronize()", "ascend_compat.streams.synchronize()"),
        # AMP
        ("torch.cuda.amp.autocast", "ascend_compat.ops.autocast"),
        ("torch.cuda.amp.GradScaler", "ascend_compat.ops.GradScaler"),
        # Seeds
        ("torch.cuda.manual_seed(", "ascend_compat.ops.manual_seed("),
        ("torch.cuda.manual_seed_all(", "ascend_compat.ops.manual_seed_all("),
    ]

    for old, new in replacements:
        if old in modified:
            modified = modified.replace(old, new)
            print(f"  Replaced: {old} → {new}")

    if not dry_run:
        path.write_text(modified, encoding="utf-8")
        print(f"  Ported file written to: {path}")

    return modified


# ---------------------------------------------------------------------------
# Info command
# ---------------------------------------------------------------------------


def show_info() -> str:
    """Show detected hardware and shim status."""
    lines = ["ascend-compat system info", "=" * 40]

    try:
        import torch
        lines.append(f"PyTorch version:   {torch.__version__}")
    except ImportError:
        lines.append("PyTorch:           NOT INSTALLED")
        return "\n".join(lines)

    try:
        import ascend_compat
        lines.append(f"ascend-compat:     {ascend_compat.__version__}")
        lines.append(f"Shim activated:    {ascend_compat.is_activated()}")
        lines.append(f"Preferred backend: {ascend_compat.preferred_backend().value}")
        lines.append(f"Has NPU:           {ascend_compat.has_npu()}")
        lines.append(f"Has CUDA:          {ascend_compat.has_cuda()}")
        backends = ascend_compat.detect_backends()
        lines.append(f"All backends:      {[b.value for b in backends]}")
    except Exception as e:
        lines.append(f"Error loading ascend_compat: {e}")

    try:
        import torch_npu  # type: ignore
        lines.append(f"torch_npu version: {torch_npu.__version__}")
    except ImportError:
        lines.append("torch_npu:         NOT INSTALLED")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _run_script(script_path: str, script_args: List[str]) -> int:
    """Launch a Python script with full ascend-compat shims active.

    This is the ``ascend-compat run`` command.  It:
    1. Sets ASCEND_COMPAT_AUTO_ACTIVATE so any ``import ascend_compat`` in the
       user script triggers activation automatically
    2. Explicitly activates the cuda_shim
    3. Installs flash_attn import hook
    4. Applies ecosystem patches
    5. Runs the user's script via runpy
    """
    import runpy

    # Set env var so if the user script does `import ascend_compat`,
    # activation happens automatically (belt-and-suspenders)
    os.environ["ASCEND_COMPAT_AUTO_ACTIVATE"] = "1"

    # Explicitly activate the shim
    from ascend_compat.cuda_shim import activate
    activate()

    # Install flash_attn hook
    from ascend_compat.ecosystem._flash_attn_hook import install_flash_attn_hook
    install_flash_attn_hook()

    # Apply ecosystem patches (safe even if libraries aren't installed)
    from ascend_compat.ecosystem import transformers_patch, deepspeed_patch, vllm_patch
    transformers_patch.apply()
    deepspeed_patch.apply()
    vllm_patch.apply()

    # Prepare sys.argv for the target script
    sys.argv = [script_path] + script_args

    try:
        runpy.run_path(script_path, run_name="__main__")
        return 0
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 0
    except Exception as e:
        print(f"Error running {script_path}: {e}", file=sys.stderr)
        return 1


def _run_doctor(full: bool = False) -> str:
    """Run the doctor diagnostic checks."""
    if full:
        from ascend_compat.doctor.env_setup import full_environment_check, format_env_report
        results = full_environment_check()
        return format_env_report(results)

    from ascend_compat.doctor.version_check import check_versions, format_report
    results = check_versions()
    return format_report(results)


def _translate_cann_error(code: str) -> str:
    """Translate a CANN error code."""
    from ascend_compat.doctor.error_codes import format_error
    return format_error(code)


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point.

    Usage::

        ascend-compat check model.py
        ascend-compat check model.py --json
        ascend-compat port model.py
        ascend-compat port model.py --dry-run
        ascend-compat doctor
        ascend-compat error 507035
        ascend-compat info
    """
    parser = argparse.ArgumentParser(
        prog="ascend-compat",
        description="CUDA → Ascend NPU migration tool",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # check command
    check_parser = subparsers.add_parser(
        "check",
        help="Scan a Python file for CUDA dependencies",
    )
    check_parser.add_argument("file", help="Python file to scan")
    check_parser.add_argument(
        "--json", action="store_true", help="Output as JSON"
    )

    # port command
    port_parser = subparsers.add_parser(
        "port",
        help="Auto-rewrite simple CUDA calls to ascend-compat",
    )
    port_parser.add_argument("file", help="Python file to port")
    port_parser.add_argument(
        "--dry-run", action="store_true", help="Show changes without writing"
    )

    # doctor command
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Run environment diagnostics (versions, hardware, compatibility)",
    )
    doctor_parser.add_argument(
        "--full", action="store_true",
        help="Run deep environment validation (CANN dirs, driver, firmware, libs)",
    )

    # error command
    error_parser = subparsers.add_parser(
        "error",
        help="Translate a CANN error code to human-readable diagnosis",
    )
    error_parser.add_argument("code", help="CANN error code (e.g. 507035, ERR99999)")

    # run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run a script with full ascend-compat shims active",
    )
    run_parser.add_argument("script", help="Python script to run")
    run_parser.add_argument(
        "script_args", nargs=argparse.REMAINDER,
        help="Arguments to pass to the script",
    )

    # quant command
    quant_parser = subparsers.add_parser(
        "quant",
        help="Check quantization compatibility for a model",
    )
    quant_parser.add_argument("model", help="Model name or path (e.g. meta-llama/Llama-3-8B-GPTQ)")

    # vllm command
    subparsers.add_parser(
        "vllm",
        help="Check vLLM/vllm-ascend readiness",
    )

    # scaffold command
    scaffold_parser = subparsers.add_parser(
        "scaffold",
        help="Generate Ascend C operator project from a template",
    )
    scaffold_parser.add_argument("name", help="Operator name (PascalCase, e.g. FusedRMSNorm)")
    scaffold_parser.add_argument(
        "--pattern", default="elementwise",
        choices=["elementwise", "reduction", "matmul", "custom"],
        help="Computation pattern (default: elementwise)",
    )
    scaffold_parser.add_argument(
        "--output", "-o", default=None,
        help="Output directory (default: ./<name_lower>_op)",
    )

    # bench command
    bench_parser = subparsers.add_parser(
        "bench",
        help="Run benchmarks (overhead measurement, op latency, memory bandwidth)",
    )
    bench_parser.add_argument(
        "mode", nargs="?", default="overhead",
        choices=["overhead", "ops", "bandwidth"],
        help="Benchmark mode: 'overhead', 'ops', or 'bandwidth'",
    )
    bench_parser.add_argument(
        "--device", default="cpu",
        help="Device for ops benchmark (default: cpu)",
    )
    bench_parser.add_argument(
        "--iterations", type=int, default=None,
        help="Number of iterations (default: 50000 for overhead, 1000 for ops)",
    )
    bench_parser.add_argument(
        "--csv", default=None, metavar="FILE",
        help="Export results to CSV file",
    )

    # compile command
    compile_parser = subparsers.add_parser(
        "compile",
        help="Show torch.compile backend info for Ascend",
    )

    # security command
    security_parser = subparsers.add_parser(
        "security",
        help="Verify torch_npu and CANN binary integrity",
    )

    # info command
    subparsers.add_parser("info", help="Show system info and shim status")

    args = parser.parse_args(argv)

    if args.command == "check":
        try:
            report = check_file(args.file)
            if args.json:
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
                print(json.dumps(data, indent=2))
            else:
                print(report.summary())
            return 0
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    elif args.command == "port":
        try:
            port_file(args.file, dry_run=args.dry_run)
            return 0
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    elif args.command == "run":
        return _run_script(args.script, args.script_args)

    elif args.command == "doctor":
        print(_run_doctor(full=args.full))
        return 0

    elif args.command == "error":
        print(_translate_cann_error(args.code))
        return 0

    elif args.command == "quant":
        from ascend_compat.cuda_shim.quantization import check_model_quant, format_quant_report
        compat = check_model_quant(args.model)
        print(format_quant_report(compat))
        return 0

    elif args.command == "vllm":
        from ascend_compat.ecosystem.vllm_patch import check_vllm_readiness
        result = check_vllm_readiness()
        icon = "[OK]" if result["ready"] else "[XX]"
        print(f"{icon} vLLM readiness: {'Ready' if result['ready'] else 'Not ready'}")
        for k, v in result["info"].items():
            print(f"  {k}: {v}")
        for issue in result["issues"]:
            print(f"  [!!] {issue}")
        return 0

    elif args.command == "scaffold":
        from ascend_compat.kernel_helper import OpSpec, scaffold
        name = args.name
        output_dir = args.output or f"./{name.lower()}_op"
        spec = OpSpec(
            name=name,
            inputs=[("x", "float16")],
            outputs=[("y", "float16")],
            pattern=args.pattern,
            description=f"Auto-generated {args.pattern} operator for Ascend NPU",
        )
        files = scaffold(spec, output_dir)
        print(f"Scaffolded Ascend C operator '{name}' → {output_dir}")
        for relpath in sorted(files.keys()):
            print(f"  {relpath}")
        return 0

    elif args.command == "bench":
        from ascend_compat.bench import ShimOverheadBench, OpLatencyBench, MemoryBandwidthBench
        if args.mode == "overhead":
            iters = args.iterations or 50000
            report = ShimOverheadBench(iterations=iters).run()
        elif args.mode == "bandwidth":
            iters = args.iterations or 50
            report = MemoryBandwidthBench(device=args.device, iterations=iters).run()
        else:
            iters = args.iterations or 1000
            report = OpLatencyBench(device=args.device, iterations=iters).run()
        print(report.report())
        if args.csv:
            with open(args.csv, "w", newline="") as f:
                f.write(report.to_csv())
            print(f"\nResults exported to {args.csv}")
        return 0

    elif args.command == "compile":
        from ascend_compat.cuda_shim.compile_helpers import get_compile_info, CompatibilityPolicy
        info = get_compile_info()
        print("torch.compile configuration for Ascend:")
        print(f"  Recommended backend:  {info['recommended_backend']}")
        print(f"  torchair available:   {info['torchair_available']}")
        backends = info.get("available_backends", [])
        print(f"  Registered backends:  {', '.join(backends) if backends else '(unknown)'}")
        # Forward compatibility check
        try:
            is_tested = CompatibilityPolicy.check_forward_compat(policy="silent")
            from ascend_compat.cuda_shim.compile_helpers import LATEST_TESTED_VERSION
            tested_str = ".".join(str(v) for v in LATEST_TESTED_VERSION)
            print(f"  Latest tested PyTorch: {tested_str}")
            print(f"  Version in range:     {'yes' if is_tested else 'no (untested version)'}")
        except Exception:
            pass
        return 0

    elif args.command == "security":
        from ascend_compat.doctor.security_check import full_security_check, format_security_report
        results = full_security_check()
        print(format_security_report(results))
        errors = sum(1 for r in results if r.status == "error")
        return 1 if errors else 0

    elif args.command == "info":
        print(show_info())
        return 0

    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
