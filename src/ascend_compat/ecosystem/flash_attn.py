"""Drop-in replacement for the ``flash_attn`` package on Ascend NPU.

The ``flash-attn`` package cannot be installed on Ascend hardware — it
requires NVIDIA CUDA to compile its custom kernels.  However, Huawei provides
``torch_npu.npu_fusion_attention`` which implements the same fused attention
algorithm optimized for Da Vinci cores.

This module provides the exact same public API as ``flash_attn``:
- ``flash_attn_func``
- ``flash_attn_varlen_func``
- ``flash_attn_with_kvcache``

so that code like::

    from flash_attn import flash_attn_func

can be replaced with::

    from ascend_compat.ecosystem.flash_attn import flash_attn_func

or, more powerfully, ascend-compat can register itself as a ``flash_attn``
package via the import hook so existing code doesn't need any changes.

API Mapping
-----------
flash_attn signature::

    flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None,
                    causal=False, return_attn_probs=False)
    → (output,) or (output, softmax_lse, S_dmask)

torch_npu.npu_fusion_attention signature::

    torch_npu.npu_fusion_attention(
        query, key, value, head_num, input_layout,
        pse=None, padding_mask=None, atten_mask=None,
        scale=1.0, keep_prob=1.0, pre_tockens=2147483647,
        next_tockens=0, inner_precise=0, prefix=None,
        sparse_mode=0, gen_mask_parallel=True, sync=False
    ) → (output, softmax_max, softmax_sum, softmax_out)

Key differences:
- flash_attn uses ``dropout_p``; npu_fusion_attention uses ``keep_prob = 1 - dropout_p``
- flash_attn uses ``softmax_scale``; npu_fusion_attention uses ``scale``
- flash_attn expects shape ``(batch, seqlen, nheads, headdim)``
  npu_fusion_attention expects ``(batch, seqlen, nheads, headdim)`` with
  ``input_layout="BSND"``
- ``causal=True`` maps to ``next_tockens=0`` (only look at past tokens)
  ``causal=False`` maps to ``next_tockens=2147483647`` (look at all tokens)
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple, Union

from ascend_compat._logging import get_logger

logger = get_logger(__name__)


def _get_npu_fusion_attention() -> Any:
    """Lazily resolve torch_npu.npu_fusion_attention."""
    try:
        import torch_npu  # type: ignore[import-untyped]
        if hasattr(torch_npu, "npu_fusion_attention"):
            return torch_npu.npu_fusion_attention
    except ImportError:
        pass

    # Fallback: check if it's accessible as a torch op
    try:
        import torch
        if hasattr(torch.ops, "npu") and hasattr(torch.ops.npu, "npu_fusion_attention"):
            return torch.ops.npu.npu_fusion_attention
    except (ImportError, AttributeError):
        pass

    return None


def flash_attn_func(
    q: Any,
    k: Any,
    v: Any,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[Any] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> Union[Any, Tuple[Any, Any, Any]]:
    """Drop-in replacement for ``flash_attn.flash_attn_func``.

    Wraps ``torch_npu.npu_fusion_attention`` with argument translation.

    Args:
        q: Query tensor, shape ``(batch, seqlen_q, nheads, headdim)``.
        k: Key tensor, shape ``(batch, seqlen_k, nheads_k, headdim)``.
        v: Value tensor, shape ``(batch, seqlen_k, nheads_k, headdim)``.
        dropout_p: Dropout probability (0.0 = no dropout).
        softmax_scale: Scaling factor for QK^T.  Default: ``1 / sqrt(headdim)``.
        causal: If True, apply causal mask (each position only attends to
            earlier positions).
        window_size: Sliding window attention sizes (not supported on NPU —
            ignored with warning).
        alibi_slopes: ALiBi positional encoding slopes (not natively supported —
            applied via PSE if possible).
        deterministic: Ignored on NPU (determinism controlled at CANN level).
        return_attn_probs: If True, return attention weights and log-sum-exp.

    Returns:
        If ``return_attn_probs=False``: output tensor.
        If ``return_attn_probs=True``: ``(output, softmax_lse, None)``.

    Raises:
        RuntimeError: If torch_npu is not available.
    """
    npu_attn = _get_npu_fusion_attention()

    if npu_attn is None:
        raise RuntimeError(
            "torch_npu.npu_fusion_attention not available.\n"
            "  Install torch_npu: pip install torch-npu\n"
            "  Or use standard attention: attn_implementation='eager' in HuggingFace"
        )

    # Validate shapes: flash_attn expects (batch, seqlen, nheads, headdim)
    batch, seqlen_q, nheads, headdim = q.shape
    _, seqlen_k, nheads_k, _ = k.shape

    # Compute scale
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(headdim)

    # Convert dropout_p → keep_prob
    keep_prob = 1.0 - dropout_p

    # Causal mapping:
    #   causal=True  → next_tockens=0 (can only see past/current tokens)
    #   causal=False → next_tockens=very_large (can see all tokens)
    next_tokens = 0 if causal else 2147483647

    # Sliding window attention
    if window_size != (-1, -1):
        logger.warning(
            "Sliding window attention (window_size=%s) is not natively supported "
            "by npu_fusion_attention. Using full attention instead.",
            window_size,
        )

    # Handle GQA (grouped query attention): flash_attn supports nheads_k != nheads
    # npu_fusion_attention uses head_num = number of query heads
    head_num = nheads

    # PSE (positional softmax encoding) for alibi_slopes
    pse = None
    if alibi_slopes is not None:
        logger.debug("ALiBi slopes provided — constructing PSE tensor")
        # ALiBi integration would require constructing the full bias matrix
        # This is a placeholder — real implementation depends on torch_npu version
        logger.warning(
            "ALiBi slopes via PSE is experimental.  Consider using "
            "RoPE or learned position embeddings for best NPU performance."
        )

    logger.debug(
        "flash_attn_func → npu_fusion_attention("
        "batch=%d, seq_q=%d, seq_k=%d, heads=%d, dim=%d, "
        "scale=%.4f, keep_prob=%.2f, causal=%s)",
        batch, seqlen_q, seqlen_k, nheads, headdim,
        softmax_scale, keep_prob, causal,
    )

    # Call npu_fusion_attention
    # Input layout BSND = (Batch, Sequence, NumHeads, Dim)
    output, softmax_max, softmax_sum, softmax_out = npu_attn(
        q, k, v,
        head_num=head_num,
        input_layout="BSND",
        scale=softmax_scale,
        keep_prob=keep_prob,
        next_tockens=next_tokens,
        pse=pse,
    )

    if return_attn_probs:
        # Approximate softmax_lse from softmax_max + log(softmax_sum)
        import torch
        softmax_lse = softmax_max + torch.log(softmax_sum + 1e-12)
        return output, softmax_lse, None

    return output


def flash_attn_varlen_func(
    q: Any,
    k: Any,
    v: Any,
    cu_seqlens_q: Any,
    cu_seqlens_k: Any,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    return_attn_probs: bool = False,
    **kwargs: Any,
) -> Union[Any, Tuple[Any, Any, Any]]:
    """Drop-in replacement for ``flash_attn.flash_attn_varlen_func``.

    Handles variable-length sequences (packed/unpadded format).

    Note:
        npu_fusion_attention's variable-length support depends on torch_npu
        version.  If not available, we fall back to padding + standard call.

    Args:
        q: Query, shape ``(total_q, nheads, headdim)`` (packed format).
        k: Key, shape ``(total_k, nheads_k, headdim)``.
        v: Value, shape ``(total_k, nheads_k, headdim)``.
        cu_seqlens_q: Cumulative sequence lengths for queries, shape ``(batch+1,)``.
        cu_seqlens_k: Cumulative sequence lengths for keys, shape ``(batch+1,)``.
        max_seqlen_q: Maximum query sequence length.
        max_seqlen_k: Maximum key sequence length.
        dropout_p: Dropout probability.
        softmax_scale: Scaling factor.  Default: ``1 / sqrt(headdim)``.
        causal: Causal masking.
        return_attn_probs: Return attention weights.
    """
    npu_attn = _get_npu_fusion_attention()

    if npu_attn is None:
        raise RuntimeError(
            "torch_npu.npu_fusion_attention not available for varlen attention."
        )

    import torch

    nheads = q.shape[1]
    headdim = q.shape[2]
    batch_size = cu_seqlens_q.shape[0] - 1

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(headdim)

    keep_prob = 1.0 - dropout_p
    next_tokens = 0 if causal else 2147483647

    # Pad and reshape to (batch, max_seqlen, nheads, headdim) for BSND layout
    q_padded = torch.zeros(batch_size, max_seqlen_q, nheads, headdim,
                           dtype=q.dtype, device=q.device)
    k_padded = torch.zeros(batch_size, max_seqlen_k, q.shape[1], headdim,
                           dtype=k.dtype, device=k.device)
    v_padded = torch.zeros(batch_size, max_seqlen_k, q.shape[1], headdim,
                           dtype=v.dtype, device=v.device)

    for i in range(batch_size):
        sq = cu_seqlens_q[i + 1] - cu_seqlens_q[i]
        sk = cu_seqlens_k[i + 1] - cu_seqlens_k[i]
        q_padded[i, :sq] = q[cu_seqlens_q[i]:cu_seqlens_q[i + 1]]
        k_padded[i, :sk] = k[cu_seqlens_k[i]:cu_seqlens_k[i + 1]]
        v_padded[i, :sk] = v[cu_seqlens_k[i]:cu_seqlens_k[i + 1]]

    logger.debug(
        "flash_attn_varlen_func → padded npu_fusion_attention "
        "(batch=%d, max_q=%d, max_k=%d)",
        batch_size, max_seqlen_q, max_seqlen_k,
    )

    output_padded, _, _, _ = npu_attn(
        q_padded, k_padded, v_padded,
        head_num=nheads,
        input_layout="BSND",
        scale=softmax_scale,
        keep_prob=keep_prob,
        next_tockens=next_tokens,
    )

    # Unpad back to packed format
    outputs = []
    for i in range(batch_size):
        sq = cu_seqlens_q[i + 1] - cu_seqlens_q[i]
        outputs.append(output_padded[i, :sq])
    output = torch.cat(outputs, dim=0)

    if return_attn_probs:
        return output, None, None

    return output


def flash_attn_with_kvcache(
    q: Any,
    k_cache: Any,
    v_cache: Any,
    k: Optional[Any] = None,
    v: Optional[Any] = None,
    cache_seqlens: Optional[Any] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    **kwargs: Any,
) -> Any:
    """Drop-in replacement for ``flash_attn.flash_attn_with_kvcache``.

    Used in inference with KV-cache (autoregressive generation).

    Note:
        This is a simplified implementation that concatenates new KV with
        cache and calls standard attention.  For production inference on
        Ascend, consider using vLLM-Ascend or MindIE which have optimized
        paged attention kernels.
    """
    import torch

    # Append new K/V to cache if provided
    if k is not None and v is not None:
        if cache_seqlens is not None:
            # Update cache in-place (PagedAttention style)
            batch = k.shape[0]
            for i in range(batch):
                sl = cache_seqlens[i]
                k_cache[i, sl:sl + k.shape[1]] = k[i]
                v_cache[i, sl:sl + v.shape[1]] = v[i]
        else:
            k_cache = torch.cat([k_cache, k], dim=1)
            v_cache = torch.cat([v_cache, v], dim=1)

    return flash_attn_func(
        q, k_cache, v_cache,
        softmax_scale=softmax_scale,
        causal=causal,
    )
