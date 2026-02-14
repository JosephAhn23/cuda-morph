# cuda-morph

**Run CUDA PyTorch code on non-NVIDIA GPUs. Two lines, zero rewrites.**

```python
import ascend_compat
ascend_compat.activate()

# Everything below is unchanged CUDA code — cuda-morph redirects it at runtime
model = AutoModelForCausalLM.from_pretrained("gpt2").cuda()
output = model.generate(**inputs)
```

## The problem

The AI ecosystem is locked to `torch.cuda`. HuggingFace, DeepSpeed, vLLM, flash-attn, and thousands of training scripts hard-code CUDA calls. On any non-NVIDIA GPU, you get:

```
RuntimeError: Torch not compiled with CUDA enabled
```

Fixing this normally means weeks of manual porting per project. cuda-morph intercepts `torch.cuda` calls at the Python level and routes them to whatever backend is actually present.

## How it works

```
YOUR CODE (unchanged)
  model.cuda(), torch.device("cuda"), torch.cuda.is_available()...
          │
          ▼
cuda-morph (runtime interception)
  torch.cuda.* → torch.npu.* / torch.xpu.* / CPU fallback
  torch.device("cuda") → torch.device("npu")
  Tensor.cuda() → Tensor.npu()
          │
          ▼
PyTorch backend dispatch (Ascend CANN / ROCm HIP / oneAPI / CPU)
```

No code generation. No recompilation. Just Python-level monkey-patching with atomic rollback if anything fails.

## Current status

> **This project is simulation-validated, not hardware-validated.**
> 460+ tests pass in CPU-fallback mode. The architecture is complete but
> nobody has run a real model on real hardware through cuda-morph yet.
> That's the gap we're trying to close.

| Backend | Hardware | Status |
|---------|----------|--------|
| **Huawei Ascend** | 910B, 310P | Full shim + ecosystem patches (flash-attn, HuggingFace, DeepSpeed, vLLM). Needs hardware validation. |
| **AMD ROCm** | MI210, MI250X, MI300X | Detection + device routing. Ecosystem patches not yet implemented. |
| **Intel XPU** | Max 1550, Flex, Arc | Detection + device routing. Ecosystem patches not yet implemented. |
| **Cambricon** | MLU370, MLU590 | Detection + device routing. Ecosystem patches not yet implemented. |

**If you have any non-NVIDIA accelerator and want to help validate, [open an issue](https://github.com/JosephAhn23/cuda-morph/issues).** One confirmed "it runs Llama on my MI300X" report is worth more than another 100 CPU-mode tests.

## Try it now (CPU fallback mode)

You can see the interception working without any special hardware:

```bash
pip install torch transformers
git clone https://github.com/JosephAhn23/cuda-morph.git
cd cuda-morph && pip install -e .
python examples/demo_cpu_fallback.py
```

This runs a real HuggingFace model through the shim in CPU-fallback mode. Every `torch.cuda` call is intercepted and safely routed. You'll see the telemetry output showing exactly which calls were redirected.

## Quick start

```bash
pip install cuda-morph
```

```bash
# Run existing scripts unchanged:
cuda-morph run train.py --epochs 10

# Or activate in code:
import ascend_compat
ascend_compat.activate()
```

```bash
# See what cuda-morph detects on your system:
cuda-morph info

# Scan a file for CUDA dependencies:
cuda-morph check model.py
```

## What the shim handles

| What | How |
|------|-----|
| `torch.cuda.is_available()` | Returns `False` on NPU (prevents NCCL misdetection) |
| `torch.device("cuda")` | Remapped to `torch.device("npu")` transparently |
| `Tensor.cuda()` / `Module.cuda()` | Redirected to `.npu()` |
| `torch.cuda.amp.autocast` | Device type `"cuda"` swapped to `"npu"` |
| `torch.cuda.synchronize()`, `empty_cache()`, etc. | Routed to `torch.npu.*` equivalents |
| `flash_attn` | Drop-in replacement using `npu_fusion_attention` |
| HuggingFace `device_map="auto"` | Patched to detect NPU |
| DeepSpeed distributed backend | HCCL registered instead of NCCL |

Full compatibility matrix: [docs/compatibility_matrix.md](docs/compatibility_matrix.md)

## CLI tools

```bash
cuda-morph check model.py         # Scan for CUDA hard-coding
cuda-morph port model.py          # Add shim activation to a file
cuda-morph doctor                 # Environment diagnostics
cuda-morph doctor --full          # Deep CANN/driver validation
cuda-morph verify --device npu    # Empirical operator verification
cuda-morph bench overhead         # Measure shim overhead (<1us per call)
cuda-morph info                   # Show all detected backends
```

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

## Prior art

- [SCALE](https://docs.scale-lang.com/) / [ZLUDA](https://github.com/vosen/ZLUDA) — AMD/Intel CUDA compatibility at the driver level. cuda-morph works at the Python/PyTorch level: higher up the stack, less performance-optimal, but zero-friction and framework-aware.
- [FlagOS/FlagScale](https://github.com/FlagOpen/FlagScale) — Full-stack AI infrastructure for Chinese domestic chips. Production-validated with 658 contributors. cuda-morph is narrower: runtime shim only, focused on zero-code migration.

## License

Apache 2.0

## See also

- [STRATEGY.md](STRATEGY.md) — Project vision and roadmap
- [VALIDATION_STATUS.md](VALIDATION_STATUS.md) — What is and isn't proven
- [MIGRATION.md](MIGRATION.md) — Upgrading from older versions
- [docs/README_zh.md](docs/README_zh.md) — 中文文档
