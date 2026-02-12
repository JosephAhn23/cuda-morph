"""Layer 3: Ecosystem-specific compatibility patches.

This package provides targeted shims for the libraries that cause the most
pain when migrating to Ascend:

- ``flash_attn``: Drop-in wrapper around torch_npu.npu_fusion_attention
- ``transformers_patch``: Fixes HuggingFace device_map="auto" on NPU
- ``deepspeed_patch``: Fixes HCCL backend + timer.py stream sync bug
- ``vllm_patch``: vLLM/vllm-ascend compatibility (custom ops, quant detection)
- ``triton_bridge``: Triton-Ascend integration helpers

Usage::

    import ascend_compat  # activates cuda_shim automatically

    # Then explicitly enable ecosystem patches you need:
    from ascend_compat.ecosystem import flash_attn      # registers as flash_attn package
    from ascend_compat.ecosystem import transformers_patch
    transformers_patch.apply()

    from ascend_compat.ecosystem import vllm_patch
    vllm_patch.apply()
"""
