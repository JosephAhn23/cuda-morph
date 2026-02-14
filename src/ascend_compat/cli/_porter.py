"""Auto-add ascend-compat shim activation to CUDA files. Used by cuda-morph port."""

from __future__ import annotations

import re
from pathlib import Path


def port_file(file_path: str, dry_run: bool = False) -> str:
    """Add ascend-compat shim activation to a CUDA-dependent Python file.

    The shim approach: rather than rewriting individual ``torch.cuda.*``
    calls (which would require AST-level rewriting to be safe), we add
    ``import ascend_compat; ascend_compat.activate()`` after the torch
    imports.  The shim then transparently redirects all ``torch.cuda``
    calls at runtime.

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
    changes_made = False

    # Add import ascend_compat + activate() after torch imports
    if "ascend_compat.activate()" not in modified:
        shim_lines = (
            "\nimport ascend_compat  # CUDA→Ascend compatibility shim"
            "\nascend_compat.activate()  # Redirect torch.cuda → torch.npu"
        )

        # Find the last torch import and add after it
        import_pattern = re.compile(r"^(import torch.*|from torch.* import.*)$", re.MULTILINE)
        matches = list(import_pattern.finditer(modified))
        if matches:
            last_match = matches[-1]
            insert_pos = last_match.end()
            modified = modified[:insert_pos] + shim_lines + modified[insert_pos:]
        else:
            # No torch imports found — add at the top
            modified = shim_lines.lstrip("\n") + "\n" + modified

        changes_made = True
        print("  Added: import ascend_compat + ascend_compat.activate()")
    elif "import ascend_compat" not in modified:
        # Has activate() but no import (unlikely but handle it)
        modified = "import ascend_compat\n" + modified
        changes_made = True
        print("  Added: import ascend_compat")
    else:
        print("  ascend_compat already activated — no changes needed")

    if not dry_run and changes_made:
        path.write_text(modified, encoding="utf-8")
        print(f"  Ported file written to: {path}")
    elif not dry_run:
        print("  No changes written")

    return modified
