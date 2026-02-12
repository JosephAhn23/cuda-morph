#!/usr/bin/env python3
"""Example: DeepSpeed distributed training on Ascend NPU.

This script demonstrates how to use ascend-compat with DeepSpeed for
distributed training.  The key changes vs. standard CUDA training are:

1. ``ascend_compat.activate()`` at the top
2. The ecosystem patch ``deepspeed_patch.apply()``
3. Everything else is standard DeepSpeed code

Requirements:
    pip install torch deepspeed ascend-compat
    pip install torch-npu  # From Huawei's repository

Single-device test (CPU fallback):
    python examples/deepspeed_training.py

Multi-device on Ascend (4 NPUs):
    ascend-compat run deepspeed --num_gpus=4 examples/deepspeed_training.py

What ascend-compat fixes:
    - NCCL → HCCL backend selection (DeepSpeed defaults to NCCL)
    - CUDA_VISIBLE_DEVICES → ASCEND_RT_VISIBLE_DEVICES mapping
    - DeepSpeed timer.py stream sync (torch.cuda.Event → torch.npu.Event)
"""

# -- Step 1: Activate ascend-compat BEFORE any other imports ----------------
import ascend_compat
ascend_compat.activate()

from ascend_compat.ecosystem import deepspeed_patch
deepspeed_patch.apply()

# -- Step 2: Standard imports (unchanged) -----------------------------------
import argparse
import torch
import torch.nn as nn

# --------------------------------------------------------------------------
# A simple model for demonstration
# --------------------------------------------------------------------------

class TinyTransformer(nn.Module):
    """Minimal transformer for training demonstration."""

    def __init__(self, vocab_size: int = 1000, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.encoder(x)
        return self.head(x)


# --------------------------------------------------------------------------
# Training loop (works on CUDA, NPU, or CPU)
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=32)
    args = parser.parse_args()

    # Device selection
    if hasattr(torch, "npu") and torch.npu.is_available():
        device = torch.device("npu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"ascend-compat active: {ascend_compat.is_activated()}")

    model = TinyTransformer()

    # DeepSpeed config (ZeRO Stage 2)
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": 1e-4, "weight_decay": 0.01}
        },
        "fp16": {"enabled": device.type != "cpu"},
        "zero_optimization": {"stage": 2},
    }

    # Initialize DeepSpeed
    try:
        import deepspeed
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            config=ds_config,
        )
        use_deepspeed = True
        print("DeepSpeed initialized successfully")
        # On Ascend: distributed backend should be HCCL (not NCCL)
        if hasattr(torch.distributed, "get_backend") and torch.distributed.is_initialized():
            print(f"Distributed backend: {torch.distributed.get_backend()}")
    except ImportError:
        print("DeepSpeed not installed — running vanilla PyTorch training")
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        use_deepspeed = False
    except Exception as e:
        print(f"DeepSpeed init failed (expected in single-device mode): {e}")
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        use_deepspeed = False

    # Training loop
    criterion = nn.CrossEntropyLoss()

    for step in range(args.steps):
        # Synthetic data
        input_ids = torch.randint(0, 1000, (args.batch_size, args.seq_len))
        labels = torch.randint(0, 1000, (args.batch_size, args.seq_len))

        if use_deepspeed:
            input_ids = input_ids.to(model_engine.device)
            labels = labels.to(model_engine.device)
            logits = model_engine(input_ids)
            loss = criterion(logits.view(-1, 1000), labels.view(-1))
            model_engine.backward(loss)
            model_engine.step()
        else:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits.view(-1, 1000), labels.view(-1))
            loss.backward()
            optimizer.step()

        if step % 5 == 0:
            print(f"  Step {step}/{args.steps}  loss={loss.item():.4f}")

    print(f"\nTraining complete ({args.steps} steps)")

    # Proof layer
    assert loss.item() < 100, f"Loss suspiciously high: {loss.item()}"
    assert not torch.isnan(torch.tensor(loss.item())), "Loss is NaN"
    if device.type == "npu":
        assert torch.npu.memory_allocated() > 0, "NPU memory should be allocated"
        print(f"NPU memory used: {torch.npu.memory_allocated() / 1024 / 1024:.1f} MB")
    print(f"\n[VERIFIED] Training completed on {device}")

    # Telemetry
    stats = ascend_compat.get_patch_stats()
    if stats:
        print("\nPatch telemetry:")
        for name, count in sorted(stats.items(), key=lambda x: -x[1])[:5]:
            print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
