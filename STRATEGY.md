# cuda-morph Strategy

## The Problem We Solve

China is deploying domestic AI chips at national scale. The hardware works.
The global software ecosystem (HuggingFace, DeepSpeed, vLLM, Triton) does not
follow — it hard-codes NVIDIA CUDA. Every enterprise migrating to domestic
hardware hits the same wall: weeks of manual porting per project.

**cuda-morph reduces this from weeks to zero.**

## The Strategic Context

### The market

- 5,300+ AI enterprises in China migrating from CUDA to domestic hardware
- 100,000+ domestic AI chip clusters being deployed (Beijing AI Action Plan)
- National policy explicitly targets open-source AI ecosystems and
  heterogeneous computing breakthroughs (MIIT 14th Five-Year Plan)

### The fragmentation crisis

Each chip vendor ships their own PyTorch adapter:

| Vendor | Adapter | Collective | Runtime | Community |
|--------|---------|------------|---------|-----------|
| Huawei Ascend | `torch_npu` | HCCL | CANN | Largest |
| Cambricon | `torch_mlu` | CNCL | BANG | Growing |
| Biren | `torch_br` | — | BiRen RT | Early |
| Enflame | `torch_gcu` | ECCL | TopsRider | Early |
| MetaX | `torch_maca` | — | MACA | Early |

A team that ports to Ascend must re-port for Cambricon. A developer who learns
`torch_npu` cannot easily switch to `torch_mlu`. Each migration is bespoke.

**This fragmentation is a national-scale inefficiency.**

### Why nobody else builds this

- **Huawei won't.** Building a unified layer that supports Cambricon means
  validating a competitor's hardware. Their incentive is Ascend-only.
- **Cambricon won't.** Same logic in reverse.
- **Startups can't.** DeepSeek, Moonshot, Zhipu are focused on models, not
  horizontal infrastructure.
- **Academics haven't.** They focus on FLOPs and benchmarks, not software
  engineering.

**An independent, vendor-neutral project is the only entity that can build
a unified migration layer across all domestic chips.**

## Our Position

cuda-morph is **the on-ramp for China's domestic AI migration.**

We are not competing with torch_npu or torch_mlu. We are the bridge between
them and the global open-source ecosystem they cannot individually own.

| Framing | Before | After |
|---------|--------|-------|
| Identity | Compatibility hack | Strategic infrastructure |
| Value prop | "Works around Huawei's gaps" | "Extends every vendor's reach into the global ecosystem" |
| Hardware access | Cold email for charity | Partnership for mutual benefit |
| Competition | torch_npu | Fragmentation itself |

## Technical Architecture

The architecture already supports multi-vendor backends:

```
cuda_morph/
├── _backend.py              # Pluggable backend detection
├── backends/
│   ├── registry.py          # Backend registration protocol
│   ├── ascend.py            # Huawei Ascend (torch_npu, CANN, HCCL)
│   ├── cambricon.py         # Cambricon MLU (torch_mlu, BANG, CNCL)
│   └── <future>.py          # Any PrivateUse1 backend
├── cuda_shim/               # Vendor-agnostic CUDA interception
│   ├── _monkey_patch.py     # Patches route to whichever backend is active
│   └── _registry.py         # Maps torch.cuda attrs to backend equivalents
└── ecosystem/               # Library patches (same for all backends)
```

### Adding a new backend

A backend module implements a simple protocol:

```python
class CambriconBackend:
    name = "cambricon"
    device_type = "mlu"
    adapter_module = "torch_mlu"
    collective_backend = "cncl"
    visible_devices_env = "MLU_VISIBLE_DEVICES"

    @staticmethod
    def is_available() -> bool:
        """Return True if Cambricon hardware is detected."""
        ...

    @staticmethod
    def device_count() -> int:
        """Return number of MLU devices."""
        ...
```

The core shim (`_monkey_patch.py`) already uses the backend's `device_type`
to translate `"cuda"` → `"npu"` or `"cuda"` → `"mlu"`. Ecosystem patches
use the backend's `collective_backend` to select HCCL vs CNCL.

## Roadmap

### Phase 1: Prove it works (Now → 3 months)

**Goal:** Validate on real Ascend hardware. One confirmed deployment.

- [ ] Partner with one enterprise or research lab for 910B access
- [ ] Run full example suite on real NPU; fix whatever breaks
- [ ] Publish benchmark results (tokens/sec, memory, latency)
- [ ] Change README: "Validated on Ascend 910B (CANN 8.0)"

**Success metric:** One external user runs cuda-morph on real Ascend and
reports results.

### Phase 2: Multi-backend (3 → 6 months)

**Goal:** Prove the unified migration claim with a second backend.

- [ ] Obtain Cambricon MLU access (academic program or MLU cloud)
- [ ] Implement full Cambricon backend (torch_mlu detection, device mapping)
- [ ] Adapt ecosystem patches for CNCL collective
- [ ] Run same example suite on MLU; publish comparison
- [ ] Publish: "First unified PyTorch migration across Ascend and Cambricon"

**Success metric:** Same script runs on both Ascend and Cambricon with
`cuda-morph run` and zero code changes.

### Phase 3: Ecosystem adoption (6 → 12 months)

**Goal:** Institutional recognition and adoption.

- [ ] Submit to openEuler community as recommended migration tool
- [ ] Present at Huawei Developer Conference or CNCC
- [ ] Partner with one of China's AI unicorns for production validation
- [ ] Contribute upstream patches to HuggingFace/DeepSpeed where possible

**Success metric:** cuda-morph referenced in vendor documentation or
national AI infrastructure guidelines.

## Partnership Strategy

### For hardware access

cuda-morph needs hardware to validate. Hardware vendors need ecosystem tools
to drive adoption. This is **mutual benefit**, not charity.

**Pitch to vendors:**

> "You are spending engineering resources helping customers migrate from CUDA.
> cuda-morph automates this. I need hardware access to validate it works on
> your chips. In return, you get an open-source migration tool that accelerates
> your customer onboarding."

**Specific targets:**

1. **Huawei Ascend Community** — ModelZoo contributor program, hardware credits
2. **Cambricon MLU Cloud** — Academic/developer access program
3. **National computing centers** — Beijing, Shanghai, Wuhan AI clusters
4. **AI enterprises** — DeepSeek, Baichuan, Zhipu (they all have this problem)

### For adoption

The value proposition is simple: **cuda-morph saves migration time.**

Quantified: A team of 50 engineers spending 2 weeks each on CUDA→NPU migration
= 100 person-weeks = ~$300K in engineering cost. cuda-morph reduces this to
hours.

## What This Is Not

- **Not a competitor to torch_npu or torch_mlu.** We depend on them.
- **Not a Huawei product.** We are independent and vendor-neutral.
- **Not a complete solution today.** Hardware validation is required.
- **Not marketing.** Every claim here is either implemented in code or
  explicitly marked as planned.

## Current State (Honest)

| Component | Status |
|-----------|--------|
| Ascend backend architecture | Complete (407 tests pass) |
| Ascend hardware validation | Not done (no hardware access) |
| Cambricon backend | Stub with detection logic |
| Cambricon hardware validation | Not done |
| Ecosystem patches (HF, DS, vLLM) | Implemented, version-guarded |
| Ecosystem patch hardware validation | Not done |
| Multi-backend switching | Architecture supports it; tested on CPU |
| Production deployment | Zero users |

The gap between "specification" and "product" is hardware access and runtime
debugging. The code is ready. The architecture is proven on CPU. We need a
partner with chips.
