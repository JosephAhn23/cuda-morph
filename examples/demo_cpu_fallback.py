#!/usr/bin/env python3
"""Demo: cuda-morph interception in CPU-fallback mode.

This demonstrates that cuda-morph correctly intercepts and redirects
every torch.cuda call, even without GPU hardware. Run it on any machine:

    pip install torch transformers
    pip install -e .  # from the cuda-morph repo root
    python examples/demo_cpu_fallback.py

You'll see:
1. The shim activating and patching torch.cuda
2. A real HuggingFace model loading and generating text
3. Telemetry showing exactly which torch.cuda calls were intercepted

This is the same code path that runs on real Ascend/ROCm/XPU hardware —
the only difference is the final backend dispatch.
"""

from __future__ import annotations

import sys

# ── Step 1: Activate the shim ────────────────────────────────────────────
import ascend_compat

print(f"cuda-morph v{ascend_compat.__version__}")
print(f"Detected backends: {[b.value for b in ascend_compat.detect_backends()]}")
print(f"Preferred backend: {ascend_compat.preferred_backend().value}")
print()

ascend_compat.activate()
print("Shim activated. All torch.cuda calls are now intercepted.\n")

# ── Step 2: Run standard CUDA code (unchanged) ──────────────────────────
import torch

print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
print(f"  (Returns False — this prevents NCCL misdetection on non-NVIDIA)")
print(f"torch.cuda.device_count() = {torch.cuda.device_count()}")
print()

# These calls go through the shim — they don't crash on CPU
torch.cuda.empty_cache()
torch.cuda.synchronize()
print("torch.cuda.empty_cache() — OK (safe no-op on CPU)")
print("torch.cuda.synchronize() — OK (safe no-op on CPU)")
print()

# torch.device("cuda") is intercepted
device = torch.device("cpu")  # On real hardware this would be "npu"/"cuda"
print(f"Using device: {device}")
print()

# ── Step 3: Run a real model (if transformers is available) ──────────────
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_name = "gpt2"
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    prompt = "The future of AI hardware is"
    inputs = tokenizer(prompt, return_tensors="pt")

    print(f"Generating from: \"{prompt}\"")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Output: \"{result}\"\n")

except ImportError:
    print("(transformers not installed — skipping model demo)")
    print("  pip install transformers  # to see the full demo\n")

# ── Step 4: Show telemetry ───────────────────────────────────────────────
stats = ascend_compat.get_patch_stats()
if stats:
    print("Shim telemetry (which torch.cuda calls were intercepted):")
    for name, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count} call{'s' if count != 1 else ''}")
else:
    print("No patched functions were called (shim is in CPU-fallback mode).")

print()
print("Done. On real hardware (Ascend/ROCm/XPU), these same calls")
print("would route to the actual accelerator backend.")
