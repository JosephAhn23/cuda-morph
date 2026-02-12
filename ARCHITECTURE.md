# ascend-compat Architecture

**Read time: 5 minutes.** This doc is the single source of truth for how the codebase works.

## The Problem (one sentence)

Existing PyTorch code hard-codes `torch.cuda.*` everywhere, and it all breaks on Huawei Ascend NPUs.

## The Solution (one sentence)

A thin Python shim that intercepts `torch.cuda` calls at runtime and routes them to `torch.npu`, layered **on top** of Huawei's official `torch_npu` backend.

## What ascend-compat is NOT

- **Not a CUDA reimplementation.** `torch_npu` handles C++/CANN dispatch.
- **Not a device abstraction layer.** We don't wrap `torch.npu` — we redirect `torch.cuda` to it.
- **Not a framework.** It's a compatibility bridge. Activate it, and existing code works.

---

## Four-Layer Architecture

```
┌─────────────────────────────────────────────────────┐
│  Layer 4: ascend_compat.doctor                      │
│  Diagnostics, error translation, security, CLI      │
├─────────────────────────────────────────────────────┤
│  Layer 3: ascend_compat.ecosystem                   │
│  HuggingFace, DeepSpeed, flash-attn, vLLM, Triton  │
├─────────────────────────────────────────────────────┤
│  Layer 2: ascend_compat.cuda_shim                   │
│  torch.cuda interception, dtype mgmt, compile,      │
│  shape bucketing, quantization                      │
├─────────────────────────────────────────────────────┤
│  Layer 1: torch_npu (Huawei's code, not ours)       │
│  C++ dispatch, CANN/ACL, PrivateUse1 backend        │
└─────────────────────────────────────────────────────┘
```

Each layer depends only on the layer below it. Never upward.

---

## Layer 2: The Core (cuda_shim/)

This is where 90% of the value lives. Here's the exact call flow:

### Activation sequence

```
ascend_compat.activate()
  │
  ├─ PatchManager.increment_ref()        # Thread-safe ref counting
  │    └─ Returns True only on 0→1       # Idempotent: 2nd call is a no-op
  │
  ├─ _check_version_compatibility()      # Warn if torch/torch_npu mismatch
  │
  ├─ PatchManager.begin_batch()          # Mark atomic checkpoint
  │
  ├─ install_import_hook()               # sys.meta_path: ensure torch_npu loads
  ├─ _patch_cuda_namespace()             # torch.cuda.X → torch.npu.X (via registry)
  ├─ _patch_torch_device()               # torch.device("cuda") → torch.device("npu")
  ├─ _patch_cuda_is_available()          # torch.cuda.is_available() → False (!)
  ├─ _patch_tensor_cuda()                # tensor.cuda() → tensor.npu()
  ├─ [version-gated patches]             # torch.amp (2.4+), torch.accelerator (2.2+)
  │
  ├─ PatchManager.commit_batch()         # Success: finalize
  │
  └─ [ON FAILURE]
       PatchManager.rollback_batch()     # Undo everything atomically
```

### The is_available() decision (CRITICAL)

```python
torch.cuda.is_available()  →  False  (on NPU)
```

**Why False?** Because torch_npu makes it return True. HuggingFace `accelerate` then selects NCCL for distributed training. NCCL doesn't exist on Ascend — HCCL does. Result: silent distributed failure. Returning False forces libraries to fall through to their NPU detection paths.

### Key files

| File | What it does | When you'd touch it |
|------|-------------|---------------------|
| `_monkey_patch.py` | All patching logic. activate/deactivate. | Adding a new torch.cuda patch |
| `_patch_manager.py` | Thread-safe patch lifecycle (RLock, refcount, atomic batches, telemetry) | Never, unless fixing the patch engine itself |
| `_registry.py` | Explicit map of `torch.cuda.X → torch.npu.Y` | Adding support for a new torch.cuda function |
| `_import_hook.py` | sys.meta_path hook for torch_npu auto-loading | Debugging import ordering issues |
| `compile_helpers.py` | torch.compile backend selection, ShapeBucketer, CompatibilityPolicy | Adding compile backend support |
| `dtype_manager.py` | Auto-substitutes unsupported dtypes (e.g. bfloat16 → float16) | Updating dtype support tables |
| `quantization.py` | Quantization method compatibility database | When Ascend adds quant support |

---

## Layer 3: Ecosystem Patches (ecosystem/)

Each file is a standalone patch for one library. Pattern is always:

```python
def apply():
    """Apply patches. Safe to call even if library not installed."""
    try:
        import the_library
    except ImportError:
        return  # Not installed, nothing to do
    # ... monkey-patch specific functions
```

| File | Library | What it fixes |
|------|---------|--------------|
| `transformers_patch.py` | HuggingFace | `device_map="auto"` on NPU, flash_attn detection |
| `deepspeed_patch.py` | DeepSpeed | HCCL backend, timer.py stream sync |
| `flash_attn.py` | flash-attn | Drop-in shim → `torch_npu.npu_fusion_attention` |
| `_flash_attn_hook.py` | flash-attn | Import hook: `import flash_attn` → our shim |
| `vllm_patch.py` | vLLM | CANN env, attention backend, quant detection |
| `triton_bridge.py` | Triton | Backend config for Ascend |

---

## Layer 4: Diagnostics (doctor/)

| File | Purpose |
|------|---------|
| `version_check.py` | Is CANN/torch_npu/PyTorch/driver compatible? |
| `error_codes.py` | Translate CANN error 507035 → "op execution failed" |
| `fallback_monitor.py` | Count ops falling back to CPU at runtime |
| `op_auditor.py` | Predict NPU op coverage for a model before deployment |
| `env_setup.py` | Validate ASCEND_HOME_PATH, libs, disk space |
| `security_check.py` | Verify torch_npu binary integrity (SHA-256) |

---

## Data Flow: What Happens When Code Runs

```python
# User's existing code (unchanged):
model.cuda()
x = torch.zeros(10, device="cuda:0")
if torch.cuda.is_available():
    ...
```

After `ascend_compat.activate()`:

1. `model.cuda()` → `_monkey_patch` intercepts → calls `model.npu()`
2. `torch.device("cuda:0")` → `_monkey_patch` intercepts → `torch.device("npu:0")`
3. `torch.cuda.is_available()` → returns `False` (intentional!)
4. `torch.cuda.device_count()` → `_registry` maps to `torch.npu.device_count()`

---

## Threading Model

- **PatchManager** owns all state. One global instance (`_manager`).
- **RLock** protects all operations (reentrant — safe from within patched functions).
- **Reference counting**: `activate()` increments, `deactivate()` decrements. Patches only revert at count=0.
- **Atomic batches**: If patch 3 of 7 fails, patches 1-2 are rolled back. No half-patched state.

---

## Module Dependency Graph (simplified)

```
ascend_compat.__init__
  ├── _backend        (zero deps — just torch detection)
  ├── _logging        (zero deps — just stdlib logging)
  └── cuda_shim
        ├── _patch_manager  (deps: _logging)
        ├── _registry       (deps: _backend)
        ├── _import_hook    (deps: _backend)
        ├── _monkey_patch   (deps: all of the above)
        ├── compile_helpers (deps: _backend, _monkey_patch._pytorch_version)
        ├── dtype_manager   (deps: _backend)
        └── quantization    (deps: _backend)
```

`ecosystem/*` and `doctor/*` depend on `cuda_shim` and `_backend`. They never depend on each other.

---

## Configuration

All behavior is controlled by environment variables (no config files):

| Variable | Effect |
|----------|--------|
| `ASCEND_COMPAT_AUTO_ACTIVATE=1` | Auto-activate on import |
| `ASCEND_COMPAT_NO_PATCH=1` | Skip all patching (debugging) |
| `ASCEND_COMPAT_LOG_LEVEL=DEBUG` | Verbose logging |
| `ASCEND_COMPAT_COMPAT_POLICY=strict` | Error on untested PyTorch versions |

---

## Testing Strategy

- **Unit tests** (`test_*.py`): Mock torch_npu, test each module in isolation
- **Integration tests** (`test_integration.py`): Real torch objects, activate/deactivate cycles
- **Stress tests** (`test_stress.py`): 1000 cycles, memory leak detection, cache eviction
- **Performance tests** (`test_performance.py`): Benchmark overhead stays < 20%
- **Fixture**: `conftest.py` auto-resets PatchManager state between every test

All tests run on CPU (no NPU needed). The shim's CPU fallback mode exercises the same code paths.

---

## Adding a New Feature (checklist)

1. Which layer does it belong to? (2=shim, 3=ecosystem, 4=doctor)
2. Add the implementation in the correct module
3. If it's a new torch.cuda mapping: add to `_registry.py`, test in `test_registry.py`
4. If it's a new ecosystem patch: follow the `apply()` pattern, test in `test_ecosystem.py`
5. Export via the appropriate `__init__.py` if public
6. Run `python -m pytest tests/ -x -q` (must be 0 failures)
7. Update this file if the architecture changed
