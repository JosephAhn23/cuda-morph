# cuda-morph

**One import. Zero code changes. Run PyTorch on any non-NVIDIA accelerator.**

```python
import cuda_morph
cuda_morph.activate()

# Your existing code. Unchanged. Runs on AMD, Intel, Ascend, Cambricon.
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B").cuda()
output = model.generate(**inputs)
```

---

## Screenshots

### NPU Hardware Detection

<p align="center">
  <img src="docs/images/npu-smi-info.png" alt="npu-smi showing Ascend 910B devices" width="700">
</p>

### Environment Diagnostics & System Info

<p align="center">
  <img src="docs/images/cuda-morph-doctor-info.png" alt="cuda-morph doctor and info output" width="700">
</p>

### Live Monitor

<p align="center">
  <img src="docs/images/cuda-morph-monitor.png" alt="cuda-morph monitor showing live NPU utilization" width="700">
</p>

---

## What this is

cuda-morph is a runtime shim that intercepts `torch.cuda` calls and routes them to whatever accelerator is actually present — AMD ROCm, Intel XPU, Huawei Ascend, Cambricon MLU, or any future PyTorch backend.

**The problem it solves:** The global AI software ecosystem is locked to NVIDIA CUDA. HuggingFace, DeepSpeed, vLLM, flash-attn, and thousands of training scripts hard-code `torch.cuda`. On any non-NVIDIA hardware, this triggers `AssertionError: Torch not compiled with CUDA enabled`. The fix requires weeks of manual porting per project.

cuda-morph eliminates this.

```
┌─────────────────────────────────────────────────────────────────┐
│  YOUR CODE (unchanged)                                          │
│  model.cuda(), torch.device("cuda"), flash_attn, DeepSpeed...  │
├─────────────────────────────────────────────────────────────────┤
│  cuda-morph                                                     │
│  Intercepts torch.cuda → routes to detected backend             │
│  Patches ecosystem libraries (HuggingFace, DeepSpeed, vLLM)    │
├──────────┬──────────┬──────────┬──────────┬─────────────────────┤
│  AMD     │  Intel   │  Ascend  │Cambricon │  (future backends)  │
│  ROCm    │  XPU     │  NPU    │  MLU     │                     │
│  HIP     │  oneAPI  │  CANN   │  BANG    │                     │
│  RCCL    │  oneCCL  │  HCCL   │  CNCL   │                     │
├──────────┴──────────┴──────────┴──────────┴─────────────────────┤
│  PyTorch backend dispatch                                       │
└─────────────────────────────────────────────────────────────────┘
```

## Prior art and positioning

**This project does not compete with [FlagOS/FlagScale](https://github.com/FlagOpen/FlagScale).** FlagOS is a state-backed, production-validated ecosystem with 658 contributors, 65 partners, and institutional support. It covers operators, compilers, parallelism frameworks, and kernel generation for 12+ domestic Chinese chips. It is excellent. Use it if it fits your needs.

cuda-morph occupies different ground:

| | FlagOS | cuda-morph |
|---|--------|-----------|
| **Scope** | Full-stack AI infra (operators, compilers, parallelism) | Runtime shim only (zero-code migration) |
| **Backends** | 12+ Chinese chips + NVIDIA | Chinese chips + AMD + Intel + future |
| **Approach** | Deep optimization (rewrite operators, generate kernels) | Thin interception (patch torch.cuda at runtime) |
| **Users** | Enterprises with engineering teams | Anyone who wants to run existing code on non-NVIDIA hardware |
| **Validation** | Production-deployed | Simulation-validated (seeking hardware partners) |
| **Goal** | China AI self-sufficiency | Break CUDA lock-in globally |

Similarly, [SCALE](https://docs.scale-lang.com/) and [ZLUDA](https://github.com/vosen/ZLUDA) provide AMD/Intel CUDA compatibility at the driver/binary level. cuda-morph works at the Python/PyTorch level — higher up the stack, less performance-optimal, but zero-friction and framework-aware.

## Supported backends

| Backend | Chip | Adapter | Status |
|---------|------|---------|--------|
| **AMD ROCm** | MI210, MI250X, MI300X | PyTorch ROCm build | Stub + detection |
| **Intel XPU** | Max 1550, Flex, Arc | `intel-extension-for-pytorch` | Stub + detection |
| **Huawei Ascend** | 910B, 310P | `torch_npu` | Architecture complete, 400+ tests |
| **Cambricon** | MLU370, MLU590 | `torch_mlu` | Stub + detection |

### What "stub + detection" means

The backend module can detect whether hardware is present and what version of the adapter is installed. The ecosystem patches (flash_attn routing, collective backend mapping, etc.) are not yet implemented for that backend. The Ascend backend has the most complete implementation.

## Quick start

```bash
pip install cuda-morph
```

```bash
# Run existing scripts unchanged:
cuda-morph run train.py --epochs 10

# Or activate in code:
import cuda_morph
cuda_morph.activate()
```

### Check what's detected

```bash
cuda-morph info       # Show detected backends and shim status
cuda-morph doctor     # Full environment diagnostics
```

See the [screenshots above](#screenshots) for example output on an Ascend 910B system.

## Ecosystem patches

| Library | What cuda-morph fixes |
|---------|----------------------|
| `flash_attn` | Routes to vendor-specific fused attention (npu_fusion_attention, CK, etc.) |
| HuggingFace `transformers` | `device_map="auto"` on non-CUDA devices |
| DeepSpeed | HCCL/CNCL/RCCL/oneCCL collective backend selection |
| vLLM | Custom op compilation, attention backend routing |
| Triton | Triton-Ascend integration, kernel pattern detection |

## CLI tools

```bash
cuda-morph check model.py    # Scan for CUDA hard-coding
cuda-morph port model.py     # Auto-rewrite simple CUDA calls
cuda-morph doctor            # Environment diagnostics
cuda-morph verify --device npu  # Empirical operator verification
cuda-morph bench overhead    # Measure shim overhead
cuda-morph info              # Show all detected backends
```

## Validation status

> cuda-morph is simulation-validated, not hardware-validated.
>
> 420+ tests pass in CPU-fallback mode. Argument mappings are based on
> vendor documentation, not empirical execution. See
> [VALIDATION_STATUS.md](VALIDATION_STATUS.md) for the full breakdown.

**We are actively seeking hardware partners.** If you have access to any
non-NVIDIA accelerator and want to help validate, please open an issue.

## Development

```bash
git clone https://github.com/JosephAhn23/cuda-morph.git
cd cuda-morph
pip install -e ".[dev]"
pytest tests/ -v                      # All tests (CPU)
pytest tests/ -v --run-hardware       # Include hardware tests
```

### Adding a new backend

```python
# src/ascend_compat/backends/myvendor.py
from ascend_compat.backends.registry import BackendInfo

class MyVendorBackend(BackendInfo):
    name = "myvendor"
    device_type = "mydev"
    adapter_module = "torch_mydev"
    collective_backend = "myccl"
    visible_devices_env = "MYDEV_VISIBLE_DEVICES"
    display_name = "MyVendor Accelerator"

    @staticmethod
    def is_available() -> bool:
        try:
            import torch_mydev
            import torch
            return torch.mydev.is_available()
        except Exception:
            return False
```

Then register it in `backends/__init__.py`. That's it.

## License

Apache 2.0

## See also

- [STRATEGY.md](STRATEGY.md) — Project vision and roadmap
- [VALIDATION_STATUS.md](VALIDATION_STATUS.md) — What is and isn't proven
- [MIGRATION.md](MIGRATION.md) — Upgrading from older versions

---

# cuda-morph (中文)

**一次导入，零代码修改，在任何非NVIDIA加速器上运行PyTorch。**

## 这是什么

cuda-morph 是一个运行时适配层，拦截 `torch.cuda` 调用并路由到实际存在的加速器 — AMD ROCm、Intel XPU、华为昇腾、寒武纪，或任何未来的 PyTorch 后端。

**它解决的问题：** 全球 AI 软件生态被锁定在 NVIDIA CUDA 上。每个非 NVIDIA 硬件上运行现有代码都需要数周的手动移植。cuda-morph 将此降为零。

## 定位

**本项目不与 [FlagOS/FlagScale](https://github.com/FlagOpen/FlagScale) 竞争。** FlagOS 是国家支持的生产级基础设施，拥有658名贡献者和65个生态合作伙伴。它是优秀的。

cuda-morph 占据不同的位置：FlagOS 专注于中国 AI 自主可控（深度优化、算子生成、编译器）。cuda-morph 专注于**全球打破 CUDA 垄断**（零代码迁移、多厂商统一接口）。

## 支持的后端

| 后端 | 硬件 | 状态 |
|------|------|------|
| AMD ROCm | MI210, MI250X, MI300X | 检测桩 |
| Intel XPU | Max 1550, Flex, Arc | 检测桩 |
| 华为昇腾 | 910B, 310P | 架构完成，400+测试 |
| 寒武纪 | MLU370, MLU590 | 检测桩 |

## 快速开始

```python
import cuda_morph
cuda_morph.activate()  # 自动检测硬件，补丁 torch.cuda

# 以下代码无需修改，自动路由到检测到的后端
model = model.cuda()
```
