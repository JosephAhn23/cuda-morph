# From CUDA to Ascend in 10 Minutes

# 从 CUDA 迁移到昇腾：10 分钟指南

---

## Overview / 概述

This guide walks you through migrating a PyTorch training script from NVIDIA CUDA to Huawei Ascend NPU using ascend-compat.

本指南帮助您使用 ascend-compat 将 PyTorch 训练脚本从 NVIDIA CUDA 迁移到华为昇腾 NPU。

---

## Step 1: Install (2 minutes) / 第一步：安装

```bash
# Install ascend-compat
pip install ascend-compat

# On your Ascend machine, also install torch_npu:
# (Match your PyTorch version — see compatibility matrix)
pip install torch-npu==2.4.0  # for PyTorch 2.4.0

# Verify your environment:
ascend-compat doctor
```

Expected output:

```
ascend-compat doctor — environment check
==================================================
  [OK] Python: Python 3.10.12
  [OK] PyTorch: PyTorch 2.4.0
  [OK] torch_npu: torch_npu 2.4.0
  [OK] CANN: CANN 8.0.RC2
  [OK] Compatibility: torch_npu 2.4.0 + PyTorch 2.4.0 — compatible
  [OK] NPU Device: 8 NPU(s) available: Ascend 910B
==================================================
  All checks passed!
```

---

## Step 2: One-line migration (1 minute) / 第二步：一行代码迁移

### Before / 迁移前

```python
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel().cuda()
x = torch.randn(32, 10, device="cuda")

torch.cuda.manual_seed(42)
torch.backends.cudnn.benchmark = True

# Training loop...
with torch.cuda.amp.autocast():
    output = model(x)
```

### After / 迁移后

```python
import ascend_compat  # ← Add this ONE line
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel().cuda()
x = torch.randn(32, 10, device="cuda")

torch.cuda.manual_seed(42)
torch.backends.cudnn.benchmark = True

# Training loop...
with torch.cuda.amp.autocast():
    output = model(x)
```

**That's it.**  The shim transparently handles:
- `torch.device("cuda")` → `torch.device("npu")`
- `model.cuda()` → `model.npu()`
- `torch.cuda.manual_seed(42)` → `torch.npu.manual_seed(42)`
- `torch.backends.cudnn.benchmark = True` → no-op (safe)

**就这样。** 适配层自动处理所有 CUDA 调用的转换。

---

## Step 3: Check compatibility (2 minutes) / 第三步：检查兼容性

```bash
ascend-compat check train.py
```

Output:

```
╔══════════════════════════════════════════════════════════════╗
║  ascend-compat migration check: train.py                   ║
╠══════════════════════════════════════════════════════════════╣
║  Total CUDA references:  8                                  ║
║  Migration difficulty:   easy                               ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  🔄 Needs wrapper (ascend-compat handles)                    ║
║  L4: torch.cuda.is_available                                 ║
║  L5: .cuda()                                                 ║
║  L6: torch.cuda.manual_seed                                  ║
║  ...                                                         ║
╚══════════════════════════════════════════════════════════════╝
```

If difficulty is **easy** or **moderate**: `import ascend_compat` handles everything.

If difficulty is **hard**: some operations need manual changes (see Step 5).

---

## Step 4: FlashAttention (2 minutes) / 第四步：FlashAttention

The `flash-attn` package cannot install on Ascend.  ascend-compat provides a drop-in replacement:

flash-attn 包无法在昇腾上安装。ascend-compat 提供了替代方案：

```python
import ascend_compat
from ascend_compat.ecosystem import transformers_patch
transformers_patch.apply()

# Now flash_attention_2 works!
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B",
    attn_implementation="flash_attention_2",
    device_map="auto",
    torch_dtype=torch.float16,
)
```

Or use the launcher for zero-code-change:

```bash
# Run any script with full shims — no code changes needed:
ascend-compat run train.py --batch-size 32
```

---

## Step 5: Handle hard cases (3 minutes) / 第五步：处理困难情况

### FP64 operations / FP64 运算

Ascend does not support FP64.  If your code uses `torch.float64`:

昇腾不支持 FP64。如果代码使用了 `torch.float64`：

```python
from ascend_compat.cuda_shim.dtype_manager import apply_dtype_policy, DTypePolicy

# Automatically substitute float64 → float32
apply_dtype_policy(DTypePolicy.WARN)
```

### CUDA Graphs / CUDA 图

CUDA Graphs don't exist on Ascend.  Use `torch.compile` with the torchair backend:

CUDA 图在昇腾上不存在。使用 `torch.compile` 配合 torchair 后端：

```python
# Before (CUDA):
# g = torch.cuda.CUDAGraph()
# with torch.cuda.graph(g):
#     output = model(x)

# After (Ascend):
model = torch.compile(model, backend="torchair")
output = model(x)
```

### Distributed Training / 分布式训练

Replace NCCL with HCCL:

将 NCCL 替换为 HCCL：

```python
import ascend_compat
from ascend_compat.ecosystem import deepspeed_patch
deepspeed_patch.apply()

# DeepSpeed will now use HCCL automatically
import deepspeed
deepspeed.init_distributed()  # Uses HCCL, not NCCL
```

Or manually:

```python
import torch.distributed as dist
dist.init_process_group(backend="hccl")  # Instead of "nccl"
```

### CANN Error Codes / CANN 错误码

When you hit a cryptic error:

遇到难以理解的错误时：

```bash
ascend-compat error 507035
# → CANN 507035: Operator execution failed — internal kernel error.
#   Likely cause: Unsupported dtype (e.g. FP64) or tensor shape doesn't
#   meet alignment requirements
#   Fix: Check input dtypes — use FP16 or FP32...
```

---

## Step 6: Monitor performance (optional) / 第六步：监控性能

### Detect CPU fallback ops / 检测 CPU 回退

```python
from ascend_compat.doctor import FallbackMonitor

monitor = FallbackMonitor()
with monitor:
    for batch in dataloader:
        output = model(batch)

print(monitor.report.summary())
# → "2 CPU fallback(s): aten::histc (47 calls), aten::_unique2 (3 calls)"
```

### Audit a model before deployment / 部署前审计

```python
from ascend_compat.doctor import audit_model
import torch

model = MyModel()
sample = torch.randn(1, 3, 224, 224)
report = audit_model(model, sample)
print(report.summary())
# → "Operator Coverage: 98.5% native, 2 CPU fallback ops"
```

---

## Common Issues / 常见问题

| Issue | Cause | Fix |
|-------|-------|-----|
| `Torch not compiled with CUDA enabled` | Missing `import ascend_compat` | Add it as first import |
| NCCL timeout in distributed training | Wrong backend selected | Use `deepspeed_patch.apply()` or `backend="hccl"` |
| FlashAttention import fails | `flash-attn` not on Ascend | Use `transformers_patch.apply()` |
| Very slow training | CPU fallback ops | Run `FallbackMonitor` to identify |
| `ERR99999 UNKNOWN` | Version mismatch | Run `ascend-compat doctor` |
| OOM despite enough memory | Memory fragmentation | `torch.npu.empty_cache()` + reduce batch size |

---

## Performance Tips / 性能建议

1. **Use FP16/BF16** — The Cube Unit is 4-8x faster at FP16 than FP32
2. **Avoid dynamic shapes** — Static shapes enable CANN graph optimization
3. **Use torch.compile** — `torch.compile(model, backend="torchair")` enables kernel fusion
4. **Batch small ops** — Ascend prefers large matrix operations over many small ones
5. **Check fallback ops** — A single CPU fallback in the critical path can destroy throughput

---

## Environment Variables / 环境变量

```bash
# See every API translation (debugging):
export ASCEND_COMPAT_LOG_LEVEL=DEBUG

# Disable auto-patching:
export ASCEND_COMPAT_NO_PATCH=1

# Control NPU visibility:
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

# CANN debug logging:
export ASCEND_GLOBAL_LOG_LEVEL=1
```

---

## What's Next / 下一步

- Join the community: [GitHub Issues](https://github.com/ascend-compat/ascend-compat/issues)
- Report incompatible operators
- Contribute mappings for new torch_npu versions
- Star the repo if it saved you migration time!
