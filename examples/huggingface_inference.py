#!/usr/bin/env python3
"""Example: HuggingFace Transformers inference on Ascend NPU.

This script demonstrates running a HuggingFace model on Ascend NPU
using ascend-compat.  The ONLY change vs. standard CUDA code is the
two-line activation block at the top.

Requirements:
    pip install torch transformers ascend-compat
    pip install torch-npu  # From Huawei's repository

On CPU (for testing without hardware):
    python examples/huggingface_inference.py
    # Runs in CPU fallback mode — same code paths, just slower.

On Ascend NPU:
    ascend-compat run examples/huggingface_inference.py
    # Or: ASCEND_COMPAT_AUTO_ACTIVATE=1 python examples/huggingface_inference.py
"""

import ascend_compat
ascend_compat.activate()  # <-- This is the only change needed

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

MODEL_NAME = "gpt2"  # Small model for demonstration; swap to any HF model
PROMPT = "The future of AI hardware is"
MAX_NEW_TOKENS = 50

# --------------------------------------------------------------------------
# Device selection (works on CUDA, NPU, or CPU — unchanged code)
# --------------------------------------------------------------------------

if torch.cuda.is_available():
    # On CUDA systems, this works normally.
    # On Ascend with ascend-compat active, torch.cuda.is_available() returns
    # False (intentionally — prevents NCCL/HCCL misdetection).
    device = torch.device("cuda")
elif hasattr(torch, "npu") and torch.npu.is_available():
    device = torch.device("npu")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
print(f"ascend-compat active: {ascend_compat.is_activated()}")
print(f"ascend-compat version: {ascend_compat.__version__}")

# --------------------------------------------------------------------------
# Load model and tokenizer
# --------------------------------------------------------------------------

print(f"\nLoading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,  # Use float16 for NPU: torch.float16
)
model = model.to(device)
model.eval()

print(f"Model loaded on {next(model.parameters()).device}")

# --------------------------------------------------------------------------
# Generate text
# --------------------------------------------------------------------------

print(f"\nPrompt: {PROMPT}")

inputs = tokenizer(PROMPT, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated: {generated}")

# --------------------------------------------------------------------------
# Proof layer — assertions that verify the script actually worked
# --------------------------------------------------------------------------

assert outputs.shape[0] == 1, f"Expected batch size 1, got {outputs.shape[0]}"
assert outputs.shape[1] <= MAX_NEW_TOKENS + inputs["input_ids"].shape[1], "Exceeded max length"
assert len(generated) > len(PROMPT), "Generated text should be longer than the prompt"

if device.type == "npu":
    # These only pass on REAL NPU hardware
    assert torch.npu.memory_allocated() > 0, "NPU memory should be allocated"
    print(f"NPU memory used: {torch.npu.memory_allocated() / 1024 / 1024:.1f} MB")

print(f"\n[VERIFIED] Inference completed on {device}")

# --------------------------------------------------------------------------
# Show patch telemetry
# --------------------------------------------------------------------------

stats = ascend_compat.get_patch_stats()
if stats:
    print("\nPatch telemetry:")
    for name, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count} calls")
else:
    print("\nNo patches were called (running on native device)")
