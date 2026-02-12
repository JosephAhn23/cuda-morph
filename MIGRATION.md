# Migrating from ascend-compat v0.2.x to v0.5.x

This guide covers breaking changes and how to update your code.

---

## TL;DR

```diff
  import ascend_compat
+ ascend_compat.activate()

  # ... rest of your code unchanged ...
```

Or use the CLI launcher (no code changes needed):

```bash
ascend-compat run your_script.py
```

---

## Breaking Change: No more auto-activation on import

### What changed

In **v0.2.x**, `import ascend_compat` immediately patched `torch.cuda`, `torch.device`, `Tensor.cuda()`, etc. This was convenient but caused problems:

1. **Library safety** — If your library imported `ascend_compat` for detection purposes, it silently patched the entire process.
2. **Test isolation** — Tests couldn't import `ascend_compat` without triggering global side effects.
3. **Debugging** — It was hard to tell when/where patches were applied.

In **v0.3.0+**, `import ascend_compat` is side-effect-free by default. You must explicitly activate.

### How to update

**If you're an application developer:**

```diff
  import ascend_compat
+ ascend_compat.activate()
  import torch

  # All your CUDA code still works unchanged
  device = torch.device("cuda")   # → "npu"
  x = torch.randn(3, 3).cuda()   # → .npu()
```

**If you prefer zero code changes**, use the CLI launcher:

```bash
# Instead of: python train.py
ascend-compat run train.py --epochs 10
```

**If you want the old import-time behavior**, set an environment variable:

```bash
export ASCEND_COMPAT_AUTO_ACTIVATE=1
python train.py  # import ascend_compat now auto-activates
```

**If you're a library maintainer:**

Do **not** call `activate()` in your library. Instead:
1. Document that your library works with ascend-compat
2. Require users to activate it themselves, or use the CLI wrapper
3. Use `ascend_compat.has_npu()` for detection without activation

```python
# Good — detection only, no side effects
import ascend_compat

if ascend_compat.has_npu():
    print("NPU available — use `ascend_compat.activate()` to enable shim")
```

---

## New Feature: Observability

v0.5.0 adds patch call counters. After activation, you can see exactly which patches are being hit:

```python
import ascend_compat
ascend_compat.activate()

# ... run your model ...

stats = ascend_compat.get_patch_stats()
for name, count in sorted(stats.items(), key=lambda x: -x[1]):
    print(f"  {name}: {count} calls")

# Example output:
#   cuda.is_available: 42
#   torch.device: 137
#   Tensor.cuda: 89
```

To reset counters between experiments:

```python
ascend_compat.reset_patch_stats()
```

---

## New Feature: Thread Safety

v0.5.0 uses reference-counted activation protected by an RLock. You can safely call `activate()` from multiple threads or multiple times:

```python
ascend_compat.activate()   # ref_count = 1
ascend_compat.activate()   # ref_count = 2 (no double-patching)
ascend_compat.deactivate() # ref_count = 1 (still active)
ascend_compat.deactivate() # ref_count = 0 (patches reverted)
```

---

## New Feature: Atomic Activation

If any patch fails during `activate()`, all previously applied patches are automatically rolled back. You'll never end up in a half-patched state.

---

## New Feature: Version Compatibility Check

`activate()` now warns if PyTorch and torch_npu have mismatched major.minor versions:

```
RuntimeWarning: PyTorch 2.5.1 and torch_npu 2.4.0 have different major.minor
versions. This will likely cause instability.
```

---

## Deprecation Warning

If you import ascend-compat but never call `activate()`, you'll see a one-time warning at process exit:

```
DeprecationWarning: ascend-compat was imported but activate() was never called.
Since v0.3.0, auto-activation on import is removed.
```

This warning is suppressed:
- When running tests (pytest detected)
- When `ASCEND_COMPAT_NO_PATCH=1` is set
- When `ASCEND_COMPAT_AUTO_ACTIVATE=1` is set (auto-activation happens)

---

## Environment Variables

| Variable | v0.2.x | v0.5.x | Notes |
|----------|--------|--------|-------|
| `ASCEND_COMPAT_AUTO_ACTIVATE` | N/A | `1` to restore old import-time behavior | New |
| `ASCEND_COMPAT_NO_PATCH` | `1` to skip | `1` to skip | Unchanged |
| `ASCEND_COMPAT_LOG_LEVEL` | `DEBUG` etc. | `DEBUG` etc. | Unchanged |

---

## Version History

| Version | Key Changes |
|---------|-------------|
| v0.5.0 | Thread-safe PatchManager, telemetry, atomic activation, version guards |
| v0.4.0 | Benchmarking, torch.compile helpers, shape bucketing |
| v0.3.1 | Side-effect-free import, version-gated patching, integration tests |
| v0.3.0 | Removed auto-activation on import |
| v0.2.0 | Ascend C kernel helpers, vLLM patch, quantization, Triton bridge |
| v0.1.0 | Initial release — flash_attn shim, HuggingFace/DeepSpeed patches |
