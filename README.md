# cuda-morph

**Zero-code CUDA → domestic AI chip migration for PyTorch**

Write once. Run on any domestic chip. No code changes.

```python
import cuda_morph
cuda_morph.activate()

# Your existing CUDA code runs on Ascend, Cambricon, or any supported backend.
# No changes below this line.
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B").cuda()
output = model.generate(**inputs)
```

---

## Why this exists

China is deploying **100,000+ domestic AI chips** across national computing clusters. Over 5,300 AI enterprises are migrating from NVIDIA CUDA to domestic hardware — Huawei Ascend, Cambricon MLU, and others.

**The hardware works. The software ecosystem doesn't follow.**

Every AI framework, every HuggingFace model, every DeepSpeed training script hard-codes `torch.cuda`. On domestic hardware, this triggers `AssertionError: Torch not compiled with CUDA enabled`. The fix requires weeks of manual porting per project, per team, per enterprise.

cuda-morph eliminates this. **Zero code changes. Instant migration.**

```
┌─────────────────────────────────────────────────────────────────┐
│  YOUR CODE (unchanged)                                          │
│  model.cuda(), torch.device("cuda"), flash_attn, DeepSpeed...  │
├─────────────────────────────────────────────────────────────────┤
│  cuda-morph                                                     │
│  Intercepts torch.cuda → routes to detected backend             │
│  Patches ecosystem libraries (HuggingFace, DeepSpeed, vLLM)    │
├────────────────┬───────────────────┬────────────────────────────┤
│  Ascend NPU    │  Cambricon MLU    │  (future backends)         │
│  torch_npu     │  torch_mlu        │                            │
│  CANN / HCCL   │  BANG / CNCL      │                            │
├────────────────┴───────────────────┴────────────────────────────┤
│  PyTorch PrivateUse1 dispatch (zero overhead)                   │
└─────────────────────────────────────────────────────────────────┘
```

## The fragmentation problem

Each domestic chip vendor ships their own PyTorch adapter:

| Vendor | Chip | Adapter | Collective | Runtime |
|--------|------|---------|------------|---------|
| Huawei | Ascend 910B | `torch_npu` | HCCL | CANN |
| Cambricon | MLU370/590 | `torch_mlu` | CNCL | BANG |
| Biren | BR100 | `torch_br` | — | BiRen RT |

A developer who learns Ascend cannot easily move to Cambricon. A team that ports to Cambricon must re-port for Ascend. **This fragmentation is a national-scale problem.**

cuda-morph solves it: **one import, one API, all domestic hardware.**

---

## Quick Start

```bash
pip install cuda-morph
```

### Run existing scripts unchanged

```bash
cuda-morph run train.py --epochs 10
```

### Or activate in code (two lines)

```python
import cuda_morph
cuda_morph.activate()   # Detect hardware, patch torch.cuda

import torch
device = torch.device("cuda")   # → torch.device("npu") on Ascend
x = torch.randn(3, 3).cuda()   # → .npu() on Ascend, .mlu() on Cambricon
model = MyModel().cuda()        # → routes to detected backend
```

### Explicit backend selection

```python
cuda_morph.activate(backend="ascend")     # Force Ascend NPU
cuda_morph.activate(backend="cambricon")  # Force Cambricon MLU
cuda_morph.activate()                     # Auto-detect (default)
```

## Supported Backends

### Ascend NPU (Huawei) — Primary

**Status:** Architecture complete. Simulation-validated (407 tests). Awaiting hardware validation.

- `torch.cuda.*` → `torch.npu.*` API translation
- flash_attn → `npu_fusion_attention` argument mapping
- HuggingFace `device_map="auto"` fix
- DeepSpeed HCCL backend selection
- vLLM attention routing
- CANN error code translation (50+ codes)
- `torch.compile` → torchair backend

### Cambricon MLU — In Development

**Status:** Backend stub with detection logic. Seeking hardware access for validation.

- `torch.cuda.*` → `torch.mlu.*` API translation
- `torch.device("cuda")` → `torch.device("mlu")`
- NCCL → CNCL collective backend mapping
- `CUDA_VISIBLE_DEVICES` → `MLU_VISIBLE_DEVICES`

### Future Backends

The architecture supports any PyTorch PrivateUse1 backend. Adding a new vendor requires:

1. A backend module in `cuda_morph/backends/<vendor>/`
2. Detection logic (is the adapter installed? is hardware present?)
3. A device-string mapping (`"cuda"` → `"<device_type>"`)
4. Collective backend mapping (NCCL → vendor equivalent)

## Ecosystem Patches

| Library | What cuda-morph fixes |
|---------|----------------------|
| `flash_attn` | Drop-in replacement wrapping vendor-specific fused attention |
| HuggingFace `transformers` | `device_map="auto"`, flash_attn_2 detection |
| DeepSpeed | HCCL/CNCL backend, timer.py stream sync, visible devices |
| vLLM | Custom op compilation, attention backend routing, quantization |
| Triton | Triton-Ascend integration, CUDA kernel pattern detection |

## CLI Tools

```bash
cuda-morph doctor           # Check versions, hardware, compatibility
cuda-morph doctor --full    # Deep validation (runtime dirs, driver, firmware)
cuda-morph error 507035     # Translate CANN error codes
cuda-morph check model.py   # Scan for CUDA dependencies
cuda-morph port model.py    # Auto-rewrite simple CUDA calls
cuda-morph verify --device npu   # Empirically verify operator correctness
cuda-morph bench overhead   # Measure shim proxy overhead
cuda-morph bench ops        # Measure operation latency
cuda-morph info             # Show detected hardware and backends
```

## Critical Design Decision: `torch.cuda.is_available()` → `False`

torch_npu's `transfer_to_npu` makes `is_available()` return `True`. This causes HuggingFace accelerate to select NCCL as the distributed backend — which doesn't exist on Ascend. HCCL is the correct backend.

By making `is_available()` return `False`, we force libraries to fall through to their NPU/MLU detection paths.

```python
torch.npu.is_available()     # Direct check (Ascend)
torch.mlu.is_available()     # Direct check (Cambricon)
cuda_morph.has_npu()         # Via shim
cuda_morph.has_mlu()         # Via shim
```

## Validation Status

> **Honest assessment:** cuda-morph is simulation-validated, not hardware-validated.
>
> The architecture, patching machinery, and 407-test suite work correctly in
> CPU-fallback mode. Argument mappings are based on vendor documentation, not
> empirical execution. See [VALIDATION_STATUS.md](VALIDATION_STATUS.md) for
> the full breakdown of what is and isn't proven.

To run hardware validation (on Ascend NPU):

```bash
pytest tests/ --run-hardware -v       # Real NPU tests
cuda-morph verify --device npu        # Operator correctness check
```

## Known Limitations

| Limitation | Impact | Status |
|-----------|--------|--------|
| **No real hardware testing** | Argument mappings may be wrong | Seeking hardware partnerships |
| **flash_attn accuracy unverified** | Output may differ from CUDA | Needs NPU numerical comparison |
| **Ecosystem patches are version-fragile** | Library updates may break patches | Version guards added |
| **Cambricon backend is a stub** | Detection only, no ecosystem patches yet | Needs MLU hardware |
| **ZeRO-3 untested** | Memory alignment may differ | Only ZeRO-2 targeted |

## Project Structure

```
src/cuda_morph/
├── __init__.py                  # Public API, opt-in activation
├── _backend.py                  # Multi-vendor hardware detection
├── backends/                    # Vendor-specific backend modules
│   ├── ascend.py                # Huawei Ascend NPU (torch_npu)
│   └── cambricon.py             # Cambricon MLU (torch_mlu)
├── cuda_shim/                   # Core CUDA interception layer
│   ├── _registry.py             # torch.cuda attribute mapping
│   ├── _monkey_patch.py         # Version-gated runtime patching
│   ├── _patch_manager.py        # Thread-safe patch lifecycle
│   ├── dtype_manager.py         # Automatic dtype substitution
│   ├── quantization.py          # Quantization compatibility
│   └── compile_helpers.py       # torch.compile + shape bucketing
├── ecosystem/                   # Library-specific patches
│   ├── flash_attn.py            # flash_attn shim
│   ├── transformers_patch.py    # HuggingFace fixes
│   ├── deepspeed_patch.py       # DeepSpeed fixes
│   ├── vllm_patch.py            # vLLM fixes
│   └── triton_bridge.py         # Triton integration
├── doctor/                      # Diagnostics
│   ├── version_check.py         # Compatibility matrix
│   ├── error_codes.py           # Error code database
│   └── env_setup.py             # Environment validation
├── validation/                  # Empirical verification
│   └── op_verifier.py           # Operator correctness harness
└── kernel_helper/               # Ascend C scaffolding (experimental)
```

## Development

```bash
git clone https://github.com/JosephAhn23/cuda-morph.git
cd cuda-morph
pip install -e ".[dev]"
python -m pytest tests/ -v                    # All tests (CPU)
python -m pytest tests/ -v --run-hardware     # Include NPU tests
```

## What this project does NOT do

- **Does not replace vendor adapters** — torch_npu, torch_mlu handle C++ dispatch
- **Does not translate CUDA kernels** — architecture is fundamentally different
- **Does not add dispatch overhead** — PrivateUse1 dispatch is zero-cost
- **Does not auto-patch on import** — activation is explicit

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Vendor adapter for your hardware:
  - Ascend: `torch_npu` + CANN toolkit
  - Cambricon: `torch_mlu` + BANG toolkit

## License

Apache 2.0

## See Also

- [STRATEGY.md](STRATEGY.md) — Multi-backend vision and roadmap
- [VALIDATION_STATUS.md](VALIDATION_STATUS.md) — What is and isn't proven
- [MIGRATION.md](MIGRATION.md) — Upgrading from older versions
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — Common issues and fixes

---

# cuda-morph (中文)

**PyTorch CUDA → 国产AI芯片 零代码迁移工具**

写一次代码，运行在任何国产芯片上。无需修改代码。

## 背景

中国正在部署超过10万颗国产AI芯片。5300多家AI企业正在从NVIDIA CUDA迁移到国产硬件。每个AI框架、每个HuggingFace模型、每个DeepSpeed训练脚本都硬编码了 `torch.cuda`。

**硬件可以工作。软件生态没有跟上。**

cuda-morph 解决这个问题：**零代码修改，即时迁移。**

## 碎片化问题

每个国产芯片厂商提供自己的PyTorch适配器：华为用 `torch_npu`，寒武纪用 `torch_mlu`，壁仞用 `torch_br`。学会一个，不能直接用另一个。

cuda-morph 统一了接口：**一次导入，一个API，所有国产硬件。**

## 快速开始

```python
import cuda_morph
cuda_morph.activate()  # 自动检测硬件，补丁 torch.cuda

# 以下代码无需修改
device = torch.device("cuda")   # → 自动路由到检测到的后端
model = model.cuda()             # → 自动路由
```

或者直接运行现有脚本：

```bash
cuda-morph run train.py --epochs 10
```

## 支持的后端

| 厂商 | 芯片 | 状态 |
|------|------|------|
| 华为 | 昇腾 910B | 架构完成，仿真验证 |
| 寒武纪 | MLU370/590 | 后端桩，开发中 |
| 壁仞 | BR100 | 规划中 |

## 验证状态

> cuda-morph 目前在CPU回退模式下通过仿真验证（407项测试）。
> 参数映射基于厂商文档，尚未在真实硬件上验证。
> 详见 [VALIDATION_STATUS.md](VALIDATION_STATUS.md)。
