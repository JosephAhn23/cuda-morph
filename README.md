# ascend-compat

**CUDA → Ascend NPU compatibility shim for PyTorch**

A thin, high-value ecosystem compatibility bridge that makes existing CUDA-assuming Python code run on Huawei Ascend NPUs — without replacing torch_npu, but fixing the ecosystem gap above it.

---

## The Problem

torch_npu already handles low-level operator dispatch via PyTorch's PrivateUse1 backend.  What's broken is the **last mile**: HuggingFace Transformers, DeepSpeed, flash-attn, vLLM, and thousands of user scripts hard-code `torch.cuda` calls, triggering `AssertionError: Torch not compiled with CUDA enabled` on Ascend hardware.

## The Solution

```python
import ascend_compat
ascend_compat.activate()  # Explicit activation — no side effects on import
```

Or run existing scripts unchanged:

```bash
ascend-compat run script.py
```

ascend-compat is a **four-layer compatibility stack** built on top of torch_npu:

```
┌─────────────────────────────────────────────────────┐
│  Layer 4: ascend_compat.doctor                      │
│  Environment validation, error translation,         │
│  diagnostics CLI, benchmarking                      │
├─────────────────────────────────────────────────────┤
│  Layer 3: ascend_compat.ecosystem                   │
│  HuggingFace, DeepSpeed, flash-attn, vLLM, Triton  │
├─────────────────────────────────────────────────────┤
│  Layer 2: ascend_compat.cuda_shim                   │
│  torch.cuda API interception + intelligent routing  │
│  torch.compile helpers, shape bucketing, dtype mgmt │
├─────────────────────────────────────────────────────┤
│  Layer 1: torch_npu (Huawei — already exists)       │
│  PrivateUse1 backend, C++ dispatch, CANN/ACL        │
└─────────────────────────────────────────────────────┘
```

## Quick Start

```bash
pip install ascend-compat
```

### Activation modes

**1. Explicit activation** (recommended for applications):

```python
import ascend_compat
ascend_compat.activate()

import torch
device = torch.device("cuda")   # → torch.device("npu")
x = torch.randn(3, 3).cuda()   # → .npu()
model = MyModel().cuda()        # → .npu()
```

**2. CLI launcher** (recommended for running existing scripts unchanged):

```bash
ascend-compat run train.py --epochs 10
```

**3. Environment variable** (opt-in auto-activation on import):

```bash
export ASCEND_COMPAT_AUTO_ACTIVATE=1
python train.py
```

`import ascend_compat` alone does NOT patch anything by default — this is intentional to prevent global side effects when imported transitively by libraries.

### FlashAttention on Ascend (flagship feature)

```python
import ascend_compat
ascend_compat.activate()

from ascend_compat.ecosystem import transformers_patch
transformers_patch.apply()

# Now flash_attention_2 works on Ascend via npu_fusion_attention:
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    attn_implementation="flash_attention_2",  # Works on Ascend!
    device_map="auto",
)
```

### torch.compile on Ascend

```python
from ascend_compat.cuda_shim.compile_helpers import get_compile_backend, ShapeBucketer

# Auto-select the right backend (torchair on NPU, inductor on CUDA)
model = torch.compile(model, backend=get_compile_backend())

# Avoid CANN recompilation on dynamic shapes
bucketer = ShapeBucketer(buckets=[128, 256, 512, 1024, 2048])
padded_input = bucketer.pad(input_tensor, dim=1)
```

### Environment diagnostics

```bash
ascend-compat doctor          # Check versions, hardware, compatibility
ascend-compat doctor --full   # Deep validation (CANN dirs, driver, firmware)
ascend-compat error 507035    # Translate cryptic CANN error codes
ascend-compat check model.py  # Scan for CUDA dependencies
ascend-compat port model.py   # Auto-rewrite simple CUDA calls
ascend-compat bench overhead  # Measure shim proxy overhead
ascend-compat bench ops       # Measure operation latency
ascend-compat compile         # Show torch.compile backend info
ascend-compat quant meta-llama/Llama-3-8B-GPTQ  # Check quantization compat
ascend-compat vllm            # Check vLLM readiness
ascend-compat scaffold FusedRMSNorm  # Generate Ascend C operator project
```

## Critical Design Decision: `torch.cuda.is_available()` → `False`

This is the single most important design decision in the project.

torch_npu's `transfer_to_npu` makes `torch.cuda.is_available()` return `True`.  This causes HuggingFace accelerate to select **NCCL** as the distributed backend.  On Ascend, NCCL doesn't exist — **HCCL** is the correct backend.  The result is a *silent* distributed training failure.

By making `is_available()` return `False`, we force libraries to fall through to their NPU detection paths.  Check NPU availability explicitly:

```python
torch.npu.is_available()     # Direct check
ascend_compat.has_npu()      # Via shim
```

## Features

### Layer 2: CUDA Shim (`ascend_compat.cuda_shim`)

- **Import hook** (`sys.meta_path`) ensures torch_npu loads before torch.cuda
- **Mapping registry** classifies every `torch.cuda` attribute as direct/adapted/unsupported
- **Version-gated monkey-patching** — patches vary by PyTorch version for robustness
  - PyTorch 2.4+: patches `torch.amp.autocast` / `torch.amp.GradScaler`
  - PyTorch 2.2+: patches `torch.accelerator` for NPU detection
- **torch.device("cuda")** → `torch.device("npu")` transparent rewrite
- **Tensor.cuda()** / **Module.cuda()** → `.npu()`
- **Dtype management** — auto-substitutes unsupported dtypes (float64→float32, bfloat16→float16)
- **Quantization compatibility** — centralized detection of supported/unsupported methods
- **torch.compile helpers** — auto-selects torchair backend, shape bucketing
- CPU-safe fallbacks for development without hardware

### Layer 3: Ecosystem Patches (`ascend_compat.ecosystem`)

| Module | What it fixes |
|--------|---------------|
| `flash_attn` | Drop-in replacement wrapping `torch_npu.npu_fusion_attention` |
| `transformers_patch` | `device_map="auto"` on NPU, flash_attn_2 detection |
| `deepspeed_patch` | HCCL backend, timer.py stream sync, `ASCEND_RT_VISIBLE_DEVICES` |
| `vllm_patch` | Custom op compilation, attention backend routing, quant detection |
| `triton_bridge` | Triton-Ascend integration, CUDA kernel pattern detection |

### Layer 4: Doctor (`ascend_compat.doctor`)

| Feature | Description |
|---------|-------------|
| Version check | Validates CANN ↔ torch_npu ↔ PyTorch compatibility matrix |
| Error translator | Decodes 50+ CANN error codes with version tags (507035, ERR99999, etc.) |
| Fallback monitor | Detects CPU-fallback ops, counts frequency, estimates impact |
| Op auditor | Profiles a model and reports NPU-native vs. CPU-fallback ops |
| Env setup | Deep environment validation (CANN dirs, driver, firmware, shared libs) |

### Benchmarking (`ascend_compat.bench`)

| Mode | What it measures |
|------|-----------------|
| `overhead` | Shim proxy cost (microseconds per redirected call) |
| `ops` | Operation latency (tensor creation, matmul, softmax, etc.) |
| `model` | End-to-end inference throughput (samples/sec, p50/p95/p99 latency) |

### Kernel Development (`ascend_compat.kernel_helper`)

Scaffolding for custom Ascend C operators:
- Declarative operator spec (`OpSpec`) with alignment enforcement
- Auto-generated boilerplate (kernel code, host wrapper, CMakeLists, README)
- Pattern library: elementwise, reduction, matmul

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ASCEND_COMPAT_AUTO_ACTIVATE` | `1` to auto-activate on `import ascend_compat` |
| `ASCEND_COMPAT_LOG_LEVEL` | `DEBUG` to see every API translation. Default: `WARNING` |
| `ASCEND_COMPAT_NO_PATCH` | `1` to disable patching entirely (even explicit calls) |
| `ASCEND_RT_VISIBLE_DEVICES` | Ascend equivalent of `CUDA_VISIBLE_DEVICES` |

## Project Structure

```
src/ascend_compat/
├── __init__.py                  # Public API, opt-in activation
├── _backend.py                  # Hardware detection (NPU > CUDA > CPU)
├── _logging.py                  # Structured logging
├── cli.py                       # CLI: check, port, doctor, bench, compile, ...
├── bench.py                     # Benchmarking framework
├── cuda_shim/                   # Layer 2: Core shim
│   ├── _registry.py             # torch.cuda → torch.npu mapping table
│   ├── _import_hook.py          # sys.meta_path interceptor
│   ├── _monkey_patch.py         # Version-gated runtime patching
│   ├── dtype_manager.py         # Automatic dtype substitution
│   ├── quantization.py          # Quantization compatibility (single source of truth)
│   └── compile_helpers.py       # torch.compile + shape bucketing
├── ecosystem/                   # Layer 3: Library-specific patches
│   ├── flash_attn.py            # flash_attn → npu_fusion_attention
│   ├── _flash_attn_hook.py      # sys.meta_path hook for flash_attn import
│   ├── transformers_patch.py    # HuggingFace fixes
│   ├── deepspeed_patch.py       # DeepSpeed HCCL + timer fixes
│   ├── vllm_patch.py            # vLLM/vllm-ascend compatibility
│   └── triton_bridge.py         # Triton-Ascend integration
├── doctor/                      # Layer 4: Diagnostics
│   ├── version_check.py         # CANN/torch_npu/PyTorch compat matrix
│   ├── error_codes.py           # CANN error code database (versioned)
│   ├── fallback_monitor.py      # CPU fallback detection
│   ├── op_auditor.py            # Operator coverage profiler
│   └── env_setup.py             # Deep environment validation
└── kernel_helper/               # Ascend C operator scaffolding
    ├── spec.py                  # OpSpec (declarative operator definition)
    └── scaffold.py              # Code generator
```

## Development

```bash
git clone https://github.com/ascend-compat/ascend-compat.git
cd ascend-compat
pip install -e ".[dev]"
python -m pytest tests/ -v
```

### Running benchmarks

```bash
# Measure shim overhead (no hardware needed)
ascend-compat bench overhead

# Measure operation latency on NPU
ascend-compat bench ops --device npu --csv results.csv

# Or from Python
from ascend_compat.bench import OpLatencyBench
report = OpLatencyBench(device="npu").run()
print(report.report())
```

### What this project does NOT do

- **Does not replace torch_npu** — all C++/CANN integration is handled by torch_npu
- **Does not translate CUDA kernels** — Da Vinci's SIMD architecture is fundamentally different
- **Does not add dispatch overhead** — PrivateUse1 dispatch is zero-overhead
- **Does not auto-patch on import** — activation is explicit for library safety

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torch_npu (for Ascend hardware; CPU fallback for development)
- CANN toolkit (on Ascend systems)

## License

Apache 2.0

---

# ascend-compat (中文)

**PyTorch CUDA → 昇腾 NPU 兼容适配层**

一个轻量级的生态兼容桥接工具，让现有的 CUDA Python 代码无需修改即可在华为昇腾 NPU 上运行。

## 核心问题

torch_npu 已经通过 PyTorch 的 PrivateUse1 后端处理了底层算子分发。但**最后一公里**仍未解决：HuggingFace Transformers、DeepSpeed、flash-attn、vLLM 以及数千个用户脚本硬编码了 `torch.cuda` 调用，在昇腾硬件上触发 `AssertionError: Torch not compiled with CUDA enabled` 错误。

## 快速开始

```python
import ascend_compat
ascend_compat.activate()  # 显式激活 — 导入不产生副作用

# 以下代码在昇腾 NPU 上自动工作：
device = torch.device("cuda")   # → torch.device("npu")
x = x.cuda()                    # → x.npu()
model = model.cuda()             # → model.npu()
```

或者直接运行现有脚本：

```bash
ascend-compat run train.py --epochs 10
```

## 激活模式

1. **显式激活**（推荐）：`ascend_compat.activate()`
2. **CLI 启动器**：`ascend-compat run script.py`
3. **环境变量**：`export ASCEND_COMPAT_AUTO_ACTIVATE=1`

`import ascend_compat` 单独导入不会修改任何全局状态 — 这是有意设计，防止作为库被传递导入时产生副作用。

## 环境诊断

```bash
ascend-compat doctor          # 检查版本兼容性
ascend-compat doctor --full   # 深度环境验证
ascend-compat error 507035    # 翻译 CANN 错误码
ascend-compat check model.py  # 扫描 CUDA 依赖
ascend-compat port model.py   # 自动改写简单 CUDA 调用
ascend-compat bench overhead  # 测量适配层开销
ascend-compat bench ops       # 测量算子延迟
ascend-compat compile         # 显示 torch.compile 后端信息
```

## 关键设计：`torch.cuda.is_available()` 返回 `False`

torch_npu 的 `transfer_to_npu` 让 `is_available()` 返回 True，导致 accelerate 选择 NCCL 后端（昇腾上不存在）。ascend-compat 返回 False，迫使库走 NPU 检测路径。

检查 NPU 可用性请使用：
```python
torch.npu.is_available()     # 直接检查
ascend_compat.has_npu()      # 通过适配层检查
```

## 新功能 (v0.3.1)

- **无副作用导入** — `import ascend_compat` 不再自动激活
- **版本感知补丁** — 根据 PyTorch 版本条件性应用补丁
- **集成测试** — 针对真实 torch 对象验证补丁机制
- **torch.compile 支持** — 自动选择 torchair 后端
- **形状分桶** — 避免 CANN 动态形状重编译
- **基准测试** — 测量适配层开销和算子延迟
- **量化兼容性** — 集中管理支持/不支持的量化方法
- **vLLM 支持** — vllm-ascend 兼容性补丁
- **Triton 桥接** — Triton-Ascend 集成检测
- **Ascend C 脚手架** — 自定义算子项目生成
