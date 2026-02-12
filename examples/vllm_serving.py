#!/usr/bin/env python3
"""Example: vLLM inference serving on Ascend NPU.

This script demonstrates how to set up vLLM for LLM serving on Ascend
NPU using ascend-compat.  It validates the environment, checks
quantization compatibility, and runs a sample inference.

Requirements:
    pip install torch vllm ascend-compat
    pip install torch-npu vllm-ascend  # From Huawei's repository

Quick test (CPU, no vLLM required):
    python examples/vllm_serving.py --check-only
    # Validates environment and quantization compatibility

Full serving on Ascend:
    ascend-compat run python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2-7B \
        --tensor-parallel-size 4

What ascend-compat fixes:
    - CUDA_VISIBLE_DEVICES → ASCEND_RT_VISIBLE_DEVICES for tensor parallelism
    - CANN environment validation for custom op compilation
    - Quantization method compatibility checking (W8A8 yes, GPTQ/AWQ no)
    - Attention backend routing to NPU-native implementation
"""

# -- Step 1: Activate ascend-compat ----------------------------------------
import ascend_compat
ascend_compat.activate()

from ascend_compat.ecosystem import vllm_patch
vllm_patch.apply()

# -- Step 2: Standard imports ----------------------------------------------
import argparse
import sys
import torch


def check_environment() -> bool:
    """Run full environment validation for vLLM on Ascend."""
    print("=" * 60)
    print("  vLLM + Ascend Environment Check")
    print("=" * 60)

    # System info
    print(f"\n  PyTorch:        {torch.__version__}")
    print(f"  ascend-compat:  {ascend_compat.__version__}")

    try:
        import torch_npu
        print(f"  torch_npu:      {torch_npu.__version__}")
    except ImportError:
        print("  torch_npu:      NOT INSTALLED")

    # NPU detection
    if hasattr(torch, "npu") and torch.npu.is_available():
        print(f"  NPU available:  Yes ({torch.npu.device_count()} device(s))")
        print(f"  NPU model:      {torch.npu.get_device_name(0)}")
    else:
        print("  NPU available:  No (will use CPU fallback)")

    # vLLM readiness
    from ascend_compat.ecosystem.vllm_patch import check_vllm_readiness
    readiness = check_vllm_readiness()

    print(f"\n  vLLM ready:     {'Yes' if readiness['ready'] else 'No'}")
    for k, v in readiness["info"].items():
        print(f"  {k}: {v}")

    if readiness["issues"]:
        print("\n  Issues:")
        for issue in readiness["issues"]:
            print(f"    [!!] {issue}")

    # Quantization compatibility
    print("\n  Quantization support:")
    from ascend_compat.cuda_shim.quantization import get_supported_methods, get_unsupported_methods
    print(f"    Supported:   {', '.join(sorted(get_supported_methods()))}")
    print(f"    Unsupported: {', '.join(sorted(get_unsupported_methods()))}")

    # Security check
    from ascend_compat.doctor.security_check import verify_torch_npu_integrity
    integrity = verify_torch_npu_integrity()
    icon = {"ok": "[OK]", "warning": "[!!]", "error": "[XX]", "unknown": "[??]"}
    print(f"\n  Security:       {icon.get(integrity.status, '?')} {integrity.message}")

    print("=" * 60)
    return readiness["ready"]


def check_model_quant(model_name: str) -> None:
    """Check if a model's quantization method is supported on Ascend."""
    from ascend_compat.cuda_shim.quantization import check_model_quant, format_quant_report
    compat = check_model_quant(model_name)
    print(format_quant_report(compat))


def run_sample_inference(model_name: str) -> None:
    """Run a sample inference using vLLM (if available)."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("vLLM not installed. Install with: pip install vllm vllm-ascend")
        print("\nTo test without vLLM, use: python examples/huggingface_inference.py")
        return

    print(f"\nLoading {model_name} via vLLM...")
    llm = LLM(model=model_name)

    prompts = [
        "The capital of France is",
        "Explain quantum computing in one sentence:",
        "def fibonacci(n):",
    ]

    params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=64)
    outputs = llm.generate(prompts, params)

    for output in outputs:
        prompt = output.prompt
        generated = output.outputs[0].text
        print(f"\n  Prompt:    {prompt}")
        print(f"  Generated: {generated[:100]}...")

    # Proof layer
    assert len(outputs) == len(prompts), "Should have one output per prompt"
    for output in outputs:
        assert len(output.outputs) > 0, "Each prompt should produce at least one output"
        assert len(output.outputs[0].text) > 0, "Generated text should be non-empty"
    print(f"\n[VERIFIED] vLLM inference produced {len(outputs)} outputs")


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM on Ascend NPU example")
    parser.add_argument("--check-only", action="store_true",
                        help="Only run environment checks, don't serve")
    parser.add_argument("--model", default="gpt2",
                        help="Model name or path (default: gpt2)")
    parser.add_argument("--check-quant", metavar="MODEL",
                        help="Check quantization compatibility for a model")
    args = parser.parse_args()

    if args.check_quant:
        check_model_quant(args.check_quant)
        return

    ready = check_environment()

    if args.check_only:
        sys.exit(0 if ready else 1)

    run_sample_inference(args.model)


if __name__ == "__main__":
    main()
