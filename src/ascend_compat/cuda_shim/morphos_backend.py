"""The ``morphos`` ``torch.compile`` backend.

Three pieces of ``ascend_compat`` already know how to answer "will this op
be slow?" and "how do I fix it?" — but until now you had to invoke them by
hand, separately, before ever calling ``torch.compile``:

- :mod:`ascend_compat.doctor.op_auditor` — the curated tables of operators
  known to fall back to CPU (or run natively but slowly) on Ascend NPU.
- :mod:`ascend_compat.cuda_shim.compile_helpers` — picks the right
  ``torch.compile`` backend for whatever hardware is actually present
  (``torchair`` on Ascend, ``inductor`` on CUDA/CPU, ``eager`` as a last
  resort).
- :mod:`ascend_compat.kernel_helper.scaffold` — generates a real
  replacement kernel project for an operator that doesn't run natively.

``morphos`` is a thin ``torch.compile`` backend that closes the loop
between them: it inspects the captured FX graph *before* compilation,
warns about any operator known to be fallback-prone or slow, points you
at ``cuda-morph scaffold <op>`` for the worst offenders, and then
delegates the actual compilation to whichever backend
:func:`~ascend_compat.cuda_shim.compile_helpers.get_compile_backend`
selects for the active hardware.

It deliberately does **not** silently swap in a generated kernel — kernel
codegen needs a real compiler toolchain and real hardware to verify, and a
black box that claims to "just fix it" without that verification would be
lying. Telling you exactly what's wrong and which tool fixes it is the
honest version of "automatic."

Usage::

    import torch
    import ascend_compat  # registers "morphos" as a torch.compile backend

    model = torch.compile(model, backend="morphos")
"""

from __future__ import annotations

from typing import Any, Callable

from ascend_compat._logging import get_logger

logger = get_logger(__name__)

#: Diagnostics from the most recent ``morphos`` compilation, for
#: programmatic inspection (e.g. in tests or a ``cuda-morph`` report).
#: Maps op basename -> {"kind": "fallback" | "slow", "count": int, "note": str}
LAST_AUDIT: dict[str, dict[str, Any]] = {}

_registered = False


def _op_qualname(node: Any) -> tuple[str, str] | None:
    """Best-effort ``(namespace, basename)`` extraction from an FX node.

    Dynamo's captured graph (before AOTAutograd decomposition) holds
    Python-level callables like ``torch.nonzero`` or method names like
    ``"index_put_"`` — not ``aten::`` strings.  A post-decomposition node
    carries an ``OpOverload`` instead (``aten.linear.default``,
    ``quantized.linear.default``).  Both are normalized to a
    ``(namespace, basename)`` pair so they can be matched against
    op_auditor's ``namespace::basename``-keyed tables *without* colliding
    same-named ops across namespaces (``quantized::linear`` is rare and
    unsupported; plain ``aten::linear`` — i.e. ``nn.Linear`` — is not).
    """
    if node.op == "call_method":
        return "aten", str(node.target)
    target = node.target
    overloadpacket = getattr(target, "overloadpacket", None)
    if overloadpacket is not None:
        parts = str(overloadpacket).split(".")
        return (parts[0], parts[-1]) if len(parts) > 1 else ("aten", parts[-1])
    name = getattr(target, "__name__", None)
    if isinstance(name, str):
        # Plain torch.* functions (torch.nonzero, torch.where, ...) are
        # thin wrappers around the aten op of the same name.
        return "aten", name
    return None


def _audit_graph(gm: Any) -> dict[str, dict[str, Any]]:
    """Classify every node in a captured graph against the known-issue tables."""
    from ascend_compat.doctor.op_auditor import _KNOWN_SLOW_OPS, _KNOWN_UNSUPPORTED_OPS

    def _split(table: Any) -> dict[str, Any]:
        return {tuple(op.split("::", 1)): v for op, v in table.items()}

    unsupported = _split({op: True for op in _KNOWN_UNSUPPORTED_OPS})
    slow = _split(_KNOWN_SLOW_OPS)

    findings: dict[str, dict[str, Any]] = {}
    for node in gm.graph.nodes:
        qualname = _op_qualname(node)
        if qualname is None:
            continue
        if qualname in unsupported:
            entry = findings.setdefault(qualname[1], {"kind": "fallback", "count": 0, "note": ""})
            entry["count"] += 1
        elif qualname in slow:
            entry = findings.setdefault(
                qualname[1], {"kind": "slow", "count": 0, "note": slow[qualname]}
            )
            entry["count"] += 1

    return findings


def _log_audit(findings: dict[str, dict[str, Any]]) -> None:
    fallback = {k: v for k, v in findings.items() if v["kind"] == "fallback"}
    slow = {k: v for k, v in findings.items() if v["kind"] == "slow"}

    if fallback:
        names = ", ".join(f"{k} ({v['count']}x)" for k, v in sorted(fallback.items()))
        logger.warning(
            "[morphos] %d op(s) will fall back to CPU: %s. "
            "Run `cuda-morph scaffold <op>` to generate a native kernel.",
            len(fallback),
            names,
        )
    if slow:
        for name, info in sorted(slow.items()):
            logger.warning(
                "[morphos] '%s' runs natively but is known to be slow (%dx): %s",
                name,
                info["count"],
                info["note"],
            )


def morphos_backend(gm: Any, example_inputs: list, **kwargs: Any) -> Callable[..., Any]:
    """The ``torch.compile`` backend registered under the name ``"morphos"``.

    Audits the graph for fallback-prone / slow operators, logs actionable
    warnings, then delegates real compilation to
    :func:`~ascend_compat.cuda_shim.compile_helpers.get_compile_backend`'s
    choice for the active hardware.
    """
    from torch._dynamo.backends.registry import lookup_backend

    from ascend_compat.cuda_shim.compile_helpers import get_compile_backend

    global LAST_AUDIT
    LAST_AUDIT = _audit_graph(gm)
    if LAST_AUDIT:
        _log_audit(LAST_AUDIT)

    delegate_name = get_compile_backend()
    if delegate_name == "morphos":
        # Defensive: never delegate to ourselves (would recurse forever).
        delegate_name = "inductor"

    logger.debug("[morphos] delegating compilation to backend=%s", delegate_name)
    delegate_fn = lookup_backend(delegate_name)
    return delegate_fn(gm, example_inputs, **kwargs)


def register_morphos_backend() -> None:
    """Register ``"morphos"`` as a valid ``torch.compile`` backend name.

    Idempotent and side-effect-free beyond the registration itself — it
    does not patch ``torch.cuda`` or anything else, it only makes the
    string ``"morphos"`` resolvable by ``torch.compile``. Safe to call at
    package import time.
    """
    global _registered
    if _registered:
        return

    try:
        from torch._dynamo.backends.registry import _COMPILER_FNS, register_backend
    except ImportError:
        logger.debug("torch._dynamo.backends.registry unavailable — skipping morphos registration")
        return

    if "morphos" not in _COMPILER_FNS:
        register_backend(compiler_fn=morphos_backend, name="morphos")
    _registered = True
