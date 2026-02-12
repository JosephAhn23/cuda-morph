"""Version-aware mapping registry: torch.cuda → torch.npu.

Every torch.cuda attribute is classified into one of three categories:

- **direct**: Identical semantics.  ``torch.cuda.X`` → ``torch.npu.X`` with
  no argument changes.  Example: ``is_available``, ``device_count``.

- **adapted**: Same concept but arguments or return values need transformation.
  Example: ``torch.cuda.amp.autocast`` → ``torch.npu.amp.autocast`` (needs
  device_type="npu" argument on newer torch_npu).

- **unsupported**: No Ascend equivalent.  Calling raises a clear error with
  a workaround suggestion.  Example: ``memory_snapshot``, ``nvtx.*``.

The registry is keyed by torch.cuda attribute name and valued with a
:class:`Mapping` descriptor that carries the target, category, and any
required transformation logic.

Why a registry?
---------------
1. Makes the mapping explicit and auditable (vs. scattered if/else chains)
2. Enables version-awareness (torch_npu 2.1 vs 2.3 have different APIs)
3. Supports the CLI ``check`` tool — it queries this registry to assess
   compatibility of user code
4. Makes it easy for contributors to add new mappings
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ascend_compat._logging import get_logger

logger = get_logger(__name__)


class MappingKind(enum.Enum):
    """How a torch.cuda attribute maps to torch.npu."""

    DIRECT = "direct"            # Identical semantics, no arg changes
    ADAPTED = "adapted"          # Needs argument or return-value transformation
    UNSUPPORTED = "unsupported"  # No Ascend equivalent


@dataclass(frozen=True)
class Mapping:
    """Descriptor for a single torch.cuda → torch.npu mapping.

    Attributes:
        cuda_name: The torch.cuda attribute name (e.g. "is_available").
        npu_name: The torch.npu attribute name (may differ from cuda_name).
        kind: Classification of the mapping.
        adapter: Optional callable that wraps the NPU function to match
            the CUDA signature.  Only used for ``ADAPTED`` mappings.
        note: Human-readable explanation of caveats or differences.
        min_torch_npu: Minimum torch_npu version where this mapping works.
    """

    cuda_name: str
    npu_name: str
    kind: MappingKind
    adapter: Optional[Callable[..., Any]] = None
    note: str = ""
    min_torch_npu: str = "2.1.0"


# ---------------------------------------------------------------------------
# The Registry
# ---------------------------------------------------------------------------

# fmt: off
_MAPPINGS: List[Mapping] = [
    # ── Device management ─────────────────────────────────────────────
    Mapping("is_available",       "is_available",       MappingKind.DIRECT),
    Mapping("device_count",       "device_count",       MappingKind.DIRECT),
    Mapping("current_device",     "current_device",     MappingKind.DIRECT),
    Mapping("set_device",         "set_device",         MappingKind.DIRECT),
    Mapping("get_device_name",    "get_device_name",    MappingKind.DIRECT),
    Mapping("get_device_properties", "get_device_properties", MappingKind.DIRECT,
            note="Property object fields may differ (no compute capability on Ascend)"),
    Mapping("synchronize",        "synchronize",        MappingKind.DIRECT),

    # ── Memory management ─────────────────────────────────────────────
    Mapping("memory_allocated",         "memory_allocated",         MappingKind.DIRECT),
    Mapping("max_memory_allocated",     "max_memory_allocated",     MappingKind.DIRECT),
    Mapping("memory_reserved",          "memory_reserved",          MappingKind.DIRECT),
    Mapping("max_memory_reserved",      "max_memory_reserved",      MappingKind.DIRECT),
    Mapping("empty_cache",              "empty_cache",              MappingKind.DIRECT),
    Mapping("reset_peak_memory_stats",  "reset_peak_memory_stats",  MappingKind.DIRECT),
    Mapping("reset_accumulated_memory_stats", "reset_accumulated_memory_stats", MappingKind.DIRECT),
    Mapping("memory_stats",             "memory_stats",             MappingKind.DIRECT,
            note="Key names in the returned dict may differ"),
    Mapping("memory_summary",           "memory_summary",           MappingKind.DIRECT,
            note="Output format differs from CUDA"),
    Mapping("mem_get_info",             "mem_get_info",             MappingKind.DIRECT,
            min_torch_npu="2.2.0"),
    Mapping("set_per_process_memory_fraction", "set_per_process_memory_fraction", MappingKind.DIRECT),
    Mapping("memory_snapshot", "memory_snapshot", MappingKind.UNSUPPORTED,
            note="No Ascend equivalent. Use ascend_compat.doctor for memory profiling."),

    # ── Streams / Events ──────────────────────────────────────────────
    Mapping("Stream",           "Stream",           MappingKind.DIRECT),
    Mapping("Event",            "Event",            MappingKind.DIRECT),
    Mapping("current_stream",   "current_stream",   MappingKind.DIRECT),
    Mapping("default_stream",   "default_stream",   MappingKind.DIRECT),
    Mapping("set_stream",       "set_stream",       MappingKind.DIRECT),

    # ── Random number generation ──────────────────────────────────────
    Mapping("manual_seed",      "manual_seed",      MappingKind.DIRECT),
    Mapping("manual_seed_all",  "manual_seed_all",  MappingKind.DIRECT),
    Mapping("seed",             "seed",             MappingKind.DIRECT),
    Mapping("initial_seed",     "initial_seed",     MappingKind.DIRECT),
    Mapping("get_rng_state",    "get_rng_state",    MappingKind.DIRECT,
            note="State format differs — not transferable between CUDA and NPU"),
    Mapping("set_rng_state",    "set_rng_state",    MappingKind.DIRECT,
            note="State format differs — not transferable between CUDA and NPU"),

    # ── AMP (Automatic Mixed Precision) ──────────────────────────────
    # These are handled specially because they're nested (torch.cuda.amp.*)
    # The monkey-patcher maps them via torch.npu.amp.*

    # ── Profiling (unsupported — use Ascend msprof) ──────────────────
    Mapping("nvtx.range_push", "nvtx.range_push", MappingKind.UNSUPPORTED,
            note="Use Ascend msprof profiler instead: `msprof --application`"),
    Mapping("nvtx.range_pop",  "nvtx.range_pop",  MappingKind.UNSUPPORTED,
            note="Use Ascend msprof profiler instead"),
    Mapping("nvtx.mark",       "nvtx.mark",       MappingKind.UNSUPPORTED,
            note="Use Ascend msprof profiler instead"),

    # ── CUDA Graphs (unsupported — use Ascend graph mode) ────────────
    Mapping("CUDAGraph",  "CUDAGraph",  MappingKind.UNSUPPORTED,
            note="Use torch.npu.set_compile_mode('graph_mode') or torch.compile with torchair"),
    Mapping("graph",      "graph",      MappingKind.UNSUPPORTED,
            note="Use torch.compile with torchair backend for graph capture"),
]
# fmt: on


# Build a lookup dict for fast access
_REGISTRY: Dict[str, Mapping] = {m.cuda_name: m for m in _MAPPINGS}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_mapping(cuda_attr: str) -> Optional[Mapping]:
    """Look up the mapping for a torch.cuda attribute.

    Args:
        cuda_attr: Attribute name (e.g. ``"is_available"``, ``"memory_allocated"``).

    Returns:
        A :class:`Mapping` if known, or None if not in the registry.
    """
    return _REGISTRY.get(cuda_attr)


def get_all_mappings() -> Dict[str, Mapping]:
    """Return the full registry as a dict.

    Returns:
        Dict mapping cuda_attr → Mapping.
    """
    return dict(_REGISTRY)


def get_direct_mappings() -> List[Mapping]:
    """Return all direct (1:1) mappings."""
    return [m for m in _MAPPINGS if m.kind == MappingKind.DIRECT]


def get_adapted_mappings() -> List[Mapping]:
    """Return all adapted (needs transformation) mappings."""
    return [m for m in _MAPPINGS if m.kind == MappingKind.ADAPTED]


def get_unsupported() -> List[Mapping]:
    """Return all unsupported mappings."""
    return [m for m in _MAPPINGS if m.kind == MappingKind.UNSUPPORTED]


def classify_attr(cuda_attr: str) -> str:
    """Classify a torch.cuda attribute for reporting.

    Returns:
        One of: ``"direct"``, ``"adapted"``, ``"unsupported"``, ``"unknown"``.
    """
    m = _REGISTRY.get(cuda_attr)
    if m is None:
        return "unknown"
    return m.kind.value
