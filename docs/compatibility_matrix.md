# CUDA → Ascend Compatibility Matrix

This document maps `torch.cuda` APIs to their Ascend NPU equivalents as
handled by ascend-compat's mapping registry (`cuda_shim._registry`).

## Status Legend

| Icon | Status | Meaning |
|------|--------|---------|
| ✅ | **Direct** | Identical semantics.  `torch.cuda.X` → `torch.npu.X` with no changes. |
| 🔄 | **Adapted** | Same concept but needs argument/return-value transformation. |
| ❌ | **Unsupported** | No Ascend equivalent.  Raises `NotImplementedError` with guidance. |

---

## Device Management

| CUDA API | Status | Ascend Equivalent | Notes |
|----------|--------|-------------------|-------|
| `torch.cuda.is_available()` | ✅ | `torch.npu.is_available()` | **Returns False on Ascend** (prevents NCCL misdetection) |
| `torch.cuda.device_count()` | ✅ | `torch.npu.device_count()` | |
| `torch.cuda.current_device()` | ✅ | `torch.npu.current_device()` | |
| `torch.cuda.set_device(n)` | ✅ | `torch.npu.set_device(n)` | |
| `torch.cuda.get_device_name()` | ✅ | `torch.npu.get_device_name()` | Returns "Ascend 910B" etc. |
| `torch.cuda.get_device_properties()` | ✅ | `torch.npu.get_device_properties()` | No `compute_capability` field |
| `torch.cuda.synchronize()` | ✅ | `torch.npu.synchronize()` | |
| `torch.device("cuda")` | ✅ | `torch.device("npu")` | Patched at `torch.device` level |
| `Tensor.cuda()` | ✅ | `Tensor.npu()` | Patched on `torch.Tensor` |
| `Module.cuda()` | ✅ | `Module.npu()` | Patched on `torch.nn.Module` |

## Memory Management

| CUDA API | Status | Ascend Equivalent | Notes |
|----------|--------|-------------------|-------|
| `torch.cuda.memory_allocated()` | ✅ | `torch.npu.memory_allocated()` | |
| `torch.cuda.max_memory_allocated()` | ✅ | `torch.npu.max_memory_allocated()` | |
| `torch.cuda.memory_reserved()` | ✅ | `torch.npu.memory_reserved()` | |
| `torch.cuda.max_memory_reserved()` | ✅ | `torch.npu.max_memory_reserved()` | |
| `torch.cuda.empty_cache()` | ✅ | `torch.npu.empty_cache()` | |
| `torch.cuda.reset_peak_memory_stats()` | ✅ | `torch.npu.reset_peak_memory_stats()` | |
| `torch.cuda.memory_stats()` | ✅ | `torch.npu.memory_stats()` | Key names may differ |
| `torch.cuda.memory_summary()` | ✅ | `torch.npu.memory_summary()` | Output format differs |
| `torch.cuda.mem_get_info()` | ✅ | `torch.npu.mem_get_info()` | torch_npu ≥ 2.2.0 |
| `torch.cuda.set_per_process_memory_fraction()` | ✅ | `torch.npu.set_per_process_memory_fraction()` | |
| `torch.cuda.memory_snapshot()` | ❌ | N/A | Use `ascend-compat doctor` |

## Streams & Events

| CUDA API | Status | Ascend Equivalent | Notes |
|----------|--------|-------------------|-------|
| `torch.cuda.Stream()` | ✅ | `torch.npu.Stream()` | |
| `torch.cuda.Event()` | ✅ | `torch.npu.Event()` | |
| `torch.cuda.current_stream()` | ✅ | `torch.npu.current_stream()` | |
| `torch.cuda.default_stream()` | ✅ | `torch.npu.default_stream()` | |
| `torch.cuda.set_stream()` | ✅ | `torch.npu.set_stream()` | |

## Random Number Generation

| CUDA API | Status | Ascend Equivalent | Notes |
|----------|--------|-------------------|-------|
| `torch.cuda.manual_seed(n)` | ✅ | `torch.npu.manual_seed(n)` | |
| `torch.cuda.manual_seed_all(n)` | ✅ | `torch.npu.manual_seed_all(n)` | |
| `torch.cuda.seed()` | ✅ | `torch.npu.seed()` | |
| `torch.cuda.initial_seed()` | ✅ | `torch.npu.initial_seed()` | |
| `torch.cuda.get_rng_state()` | ✅ | `torch.npu.get_rng_state()` | State not transferable across backends |
| `torch.cuda.set_rng_state()` | ✅ | `torch.npu.set_rng_state()` | State not transferable across backends |

## CUDA Graphs & Profiling

| CUDA API | Status | Ascend Equivalent | Notes |
|----------|--------|-------------------|-------|
| `torch.cuda.CUDAGraph` | ❌ | N/A | Use `torch.compile` with torchair backend |
| `torch.cuda.graph` | ❌ | N/A | Use `torch.compile` with torchair |
| `torch.cuda.nvtx.*` | ❌ | N/A | Use Ascend `msprof` profiler |

## Ecosystem Compatibility

| Library | Issue | ascend-compat Fix |
|---------|-------|-------------------|
| **flash-attn** | Cannot install on Ascend | `ecosystem.flash_attn` wraps `npu_fusion_attention` |
| **HuggingFace Transformers** | `device_map="auto"` crashes | `ecosystem.transformers_patch` fixes device detection |
| **HuggingFace accelerate** | Selects NCCL instead of HCCL | `cuda_shim` returns `is_available()=False` |
| **DeepSpeed** | NCCL init fails; timer.py crashes | `ecosystem.deepspeed_patch` registers HCCL |

## Hardware Limitations (not solvable by software)

| Limitation | Details |
|------------|---------|
| No FP64 support | Ascend 910A Cube Unit only supports FP16 GEMM |
| No Triton backend | Use TorchAir/torchair instead of torch.compile+Triton |
| 16-aligned shapes | Matrix dims must be multiples of 16 for Cube Unit |
| NC1HWC0 format | Internal memory layout differs from NCHW |
| Single PrivateUse1 | Only one custom backend per process |
