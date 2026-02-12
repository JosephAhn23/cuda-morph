# cuda-morph Strategy

## One sentence

cuda-morph is a unified runtime shim that lets existing PyTorch/CUDA code
run on any non-NVIDIA accelerator with zero code changes.

## The problem

The global AI software ecosystem is locked to NVIDIA CUDA. This is not
a China-specific problem. It affects:

- **China:** 5,300+ AI enterprises migrating to domestic chips (Ascend,
  Cambricon, Biren, MooreThreads) due to export controls
- **AMD users:** flash-attn won't compile, libraries check
  `torch.version.cuda`, RCCL differs from NCCL
- **Intel users:** XPU device type is unknown to most libraries,
  `device_map="auto"` ignores Intel GPUs
- **Cloud providers:** Multi-vendor GPU pools need workload portability
- **Researchers:** Want to run on whatever hardware is available

Every non-NVIDIA accelerator has its own PyTorch adapter. Each breaks the
ecosystem in the same way. The fix is always the same: intercept `torch.cuda`
calls and route to the real device.

**Nobody builds this as a unified tool.** Each vendor builds their own
partial solution. Each solution only covers their hardware.

## What exists (and why it's not enough)

| Project | What it does | What it doesn't do |
|---------|-------------|-------------------|
| **FlagOS/FlagScale** | Full-stack AI infra for 12+ Chinese chips. 658 contributors, 65 partners, production-deployed. | Not global. AMD/Intel are not first-class. Not "zero code change" — requires integration. |
| **SCALE** | Binary-level CUDA→ROCm translation | AMD only. No Intel, no Chinese chips. Not framework-aware. |
| **ZLUDA** | CUDA→ROCm/Level Zero binary shim | AMD + Intel only. No Chinese chips. Binary level, not Python. |
| **AdaptiveCpp** | Multi-vendor via SYCL | Requires code rewrite. Not a drop-in shim. |
| **torch_npu** | Ascend PrivateUse1 backend | Ascend only. Doesn't fix ecosystem (HF, DS, vLLM). |
| **torch_mlu** | Cambricon PrivateUse1 backend | Cambricon only. Same ecosystem gap. |
| **IPEX** | Intel XPU backend | Intel only. Same ecosystem gap. |

**The gap:** No unified, framework-aware, zero-code-change runtime shim
that covers ALL non-NVIDIA accelerators.

## cuda-morph's position

```
                        Scope
                        ↑
  FlagOS ●              |  Full stack (operators, compiler, parallelism)
                        |
                        |
                        |
                        |  Runtime shim (intercept + route)
  cuda-morph ●──────────|──────────────────────────●
                        |
  SCALE/ZLUDA ●         |  Binary translation
                        |
                        └──────────────────────────→ Vendor coverage
                    China-only              ALL non-NVIDIA
```

cuda-morph is the only project targeting the (runtime shim) x (all vendors)
quadrant. This is the position.

## Architecture

```
backends/
├── registry.py          # BackendInfo protocol
├── ascend.py            # Huawei Ascend NPU (torch_npu)
├── cambricon.py         # Cambricon MLU (torch_mlu)
├── rocm.py              # AMD ROCm (HIP, PyTorch ROCm build)
├── intel.py             # Intel XPU (IPEX, Level Zero)
└── <future>.py          # Biren, MooreThreads, ARM, etc.
```

Adding a new backend = one Python file (~80 lines). Detection logic,
device type mapping, collective backend name, env var mapping. No C++.

The core shim (`cuda_shim/`) is vendor-agnostic. It reads the active
backend's `device_type` to know what to translate `"cuda"` to.

## Roadmap

### Phase 1: Validate ONE backend (now → 3 months)

**Goal:** First end-to-end proof on real hardware.

Priority: **AMD ROCm.** Reasons:
- Hardware is accessible (cloud instances, developer grants)
- Large user community (MI300X is being deployed at scale)
- Highest value gap (flash-attn doesn't compile on ROCm)
- Proves global positioning (not just a China project)

Actions:
- [ ] Apply for AMD developer hardware grant
- [ ] Run BERT + Llama inference through cuda-morph on ROCm
- [ ] Fix flash-attn routing (use Triton flash-attn or CK on ROCm)
- [ ] Publish benchmark: tokens/sec, memory, latency vs NVIDIA baseline
- [ ] README: "Validated on AMD MI300X"

Parallel track: If Ascend hardware becomes available first, validate that.
The first validated backend — whichever it is — proves the architecture.

### Phase 2: Validate SECOND backend (3 → 6 months)

**Goal:** Prove multi-vendor claim with two validated backends.

- [ ] Second backend (Ascend or Cambricon, whichever is accessible)
- [ ] Same script runs on both backends with zero code changes
- [ ] Publish: "One import, two vendors, zero rewrites"

### Phase 3: Community and adoption (6 → 12 months)

- [ ] PyPI release: `pip install cuda-morph`
- [ ] Contributor guide for adding new backends
- [ ] Partner with one enterprise or research lab for production validation
- [ ] Present at relevant conference (AMD DevDay, Huawei HDC, or PyTorch conf)

### Phase 4 (future): CPU orchestration layer

Separate project. Not cuda-morph v2. See the bottom of this document.

## Why this is defensible

1. **Vendor neutrality.** Huawei won't build AMD support. AMD won't build
   Ascend support. Only an independent project can cover all backends.

2. **Network effects.** Each new backend makes the project more valuable
   to all users. The first multi-vendor runtime shim that works will
   accumulate users faster than vendor-specific tools.

3. **Ecosystem patches are the moat.** The hard part is not device string
   translation. It's knowing that HuggingFace `accelerate` selects NCCL
   when `is_available()` returns True, and that DeepSpeed's `timer.py`
   calls `torch.cuda.Event()`. These fixes apply to ALL non-NVIDIA backends.
   Write once, benefit everyone.

## What this is NOT

- **Not a competitor to FlagOS.** FlagOS is excellent. We link to it
  in the README. Different scope, different audience.
- **Not a competitor to vendor adapters.** We depend on torch_npu,
  torch_mlu, IPEX, ROCm builds. We don't replace them.
- **Not production-ready.** Zero hardware validation. The architecture
  is correct and tested on CPU simulation. We are seeking partners.

## Current state (honest)

| Component | Status |
|-----------|--------|
| Backend registry (4 vendors) | Complete |
| Ascend backend architecture | Complete (420+ tests) |
| AMD ROCm backend | Stub (detection only) |
| Intel XPU backend | Stub (detection only) |
| Cambricon backend | Stub (detection only) |
| Hardware validation (any backend) | **Not done** |
| Production users | **Zero** |

The gap between "specification" and "product" is access to one non-NVIDIA
accelerator for two weeks of runtime debugging.

## Future: CPU orchestration layer

Alibaba's Jade system proved that CPU-as-commander architecture delivers
22% p95 latency reduction and 33% lower power across 32,000 nodes. Their
solution is closed-source and Alibaba-internal.

No open-source alternative exists. This is a real gap (cited by 人民网).

This is a separate project — too large for cuda-morph, requires systems
engineering expertise, and should be community-owned from day one. The
right time to start is after cuda-morph has at least one validated backend
and a contributor community. Plant the flag with an RFC, not code.
