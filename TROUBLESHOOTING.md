# Troubleshooting ascend-compat

Common issues and their solutions. Run `ascend-compat doctor --full` first to get a full diagnostic report.

---

## Environment & Setup

### "ImportError: libascendcl.so cannot open shared object file"

**Cause:** CANN toolkit's shared libraries are not on the library path.

**Fix:**
```bash
# Add CANN libraries to LD_LIBRARY_PATH
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

# Or manually:
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH
```

**Verify:** `ls /usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so`

### "ImportError: No module named 'torch_npu'"

**Cause:** torch_npu is not installed, or installed for a different Python.

**Fix:**
```bash
# Install from Huawei's repository (match your PyTorch version)
pip install torch-npu==2.5.1  # For PyTorch 2.5.1

# Verify:
python -c "import torch_npu; print(torch_npu.__version__)"
```

### "RuntimeWarning: PyTorch X.Y.Z and torch_npu A.B.C have different major.minor versions"

**Cause:** Version mismatch between PyTorch and torch_npu.

**Fix:** Install matching versions. torch_npu version must match PyTorch major.minor:
```bash
# PyTorch 2.5.x → torch_npu 2.5.x
pip install torch==2.5.1 torch-npu==2.5.1
```

**Check:** `ascend-compat doctor`

---

## Activation & Patching

### "DeprecationWarning: ascend-compat was imported but activate() was never called"

**Cause:** Since v0.3.0, `import ascend_compat` no longer auto-activates.

**Fix (choose one):**
```python
# Option 1: Explicit activation
import ascend_compat
ascend_compat.activate()

# Option 2: Environment variable
# export ASCEND_COMPAT_AUTO_ACTIVATE=1

# Option 3: CLI launcher
# ascend-compat run your_script.py
```

See [MIGRATION.md](MIGRATION.md) for details.

### "AssertionError: Torch not compiled with CUDA enabled"

**Cause:** Code calls `torch.cuda` functions without ascend-compat active.

**Fix:** Make sure `ascend_compat.activate()` is called **before** any `torch.cuda` calls:
```python
import ascend_compat
ascend_compat.activate()  # Must be before torch.cuda usage

import torch
# Now torch.cuda calls are redirected to NPU
```

### Patches seem to have no effect

**Check 1:** Is the shim actually active?
```python
import ascend_compat
print(ascend_compat.is_activated())  # Should be True
```

**Check 2:** Is `ASCEND_COMPAT_NO_PATCH=1` set?
```bash
echo $ASCEND_COMPAT_NO_PATCH  # Should be empty
```

**Check 3:** Look at patch stats:
```python
stats = ascend_compat.get_patch_stats()
print(stats)  # Shows which patches are being called
```

---

## Performance Issues

### NPU performance is much slower than expected

**Run diagnostics:**
```bash
ascend-compat bench ops --device npu
ascend-compat doctor --full
```

**Common causes:**

1. **CPU fallback ops** — Check for warnings like "operator will fall back to CPU":
   ```python
   from ascend_compat.doctor.fallback_monitor import FallbackMonitor
   with FallbackMonitor() as mon:
       model(input)
   print(mon.report_summary())
   ```

2. **Graph recompilation** — Dynamic shapes cause CANN to recompile:
   ```python
   from ascend_compat.cuda_shim.compile_helpers import ShapeBucketer
   bucketer = ShapeBucketer(buckets=[128, 256, 512, 1024])
   padded = bucketer.pad(input_tensor, dim=1)
   ```

3. **FP32 on Ascend 910A** — The Cube Unit only supports FP16 matmul. Use mixed precision:
   ```python
   with torch.amp.autocast("npu", dtype=torch.float16):
       output = model(input)
   ```

4. **PCIe bandwidth** — Host-to-device copies are slow on PCIe Gen3. Minimize `.cpu()` calls.

5. **Power capping** — Check `npu-smi info` for power limits.

### ShapeBucketer memory grows unbounded

**Cause:** Using `pad_cached()` without cache limits or cleanup.

**Fix:**
```python
# Set max cache entries
bucketer = ShapeBucketer(max_cache_entries=100)

# Clear cache between epochs
bucketer.clear_cache()

# Monitor cache memory
print(f"Cache using {bucketer.cache_memory_bytes() / 1e6:.1f} MB")
```

### Benchmark results differ between runs

**Cause:** Thermal throttling, background processes, or insufficient warmup.

**Fix:** Use more iterations and check system fingerprint:
```python
from ascend_compat.bench import get_system_fingerprint
print(get_system_fingerprint())  # Compare across runs
```

---

## torch.compile Issues

### "torch.compile failed, falling back to eager mode"

**Run diagnostics:**
```bash
ascend-compat compile  # Shows available backends
```

**Common fixes:**

1. **torchair not installed:**
   ```bash
   pip install torchair  # From Huawei's repo
   ```

2. **Dynamic shapes:** Use `ShapeBucketer` or `torch.compile(dynamic=False)`.

3. **Unsupported ops:** Some custom operators may not have torchair lowerings.
   Use `safe_compile()` for automatic fallback:
   ```python
   from ascend_compat.cuda_shim.compile_helpers import safe_compile
   model = safe_compile(model)  # Falls back gracefully
   ```

### "AITemplate not found" / "Cannot find Ascend compilation tools"

**Fix:**
```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
export ASCEND_COMPAT_COMPILE_BACKEND=eager  # Or use safe_compile()
```

---

## CANN Error Codes

### CANN error 507035

**Meaning:** Operator execution failed.

**Fix:** `ascend-compat error 507035` for full details.

### CANN error 507008

**Meaning:** Operator not found in the op library.

**Fix:** Check your CANN version supports this op. Update CANN if needed.

### ERR99999 / error 99999

**Meaning:** Unknown internal error (catch-all).

**Fix:**
1. Check CANN logs: `cat /var/log/npu/slog/host-0/*.log | tail -100`
2. Update CANN and torch_npu to latest
3. Report to Huawei with `ascend-compat doctor --full` output

---

## Quantization Issues

### "Quantization method 'X' is NOT supported on Ascend NPU"

**Check compatibility:**
```bash
ascend-compat quant your-model-name
```

**Supported on Ascend 910B:** W8A8, W8A8-dynamic, SmoothQuant
**NOT supported:** FP8, AWQ, GPTQ, bitsandbytes, Marlin

**Migration path for unsupported methods:**
- GPTQ/AWQ → W8A8 (re-quantize with Ascend-compatible tools)
- FP8 → W8A8 or float16 (FP8 requires Ascend 950 hardware)
- bitsandbytes → SmoothQuant

---

## Distributed Training

### NCCL errors on Ascend

**Cause:** NCCL is an NVIDIA-specific library. Ascend uses HCCL.

**Fix:** ascend-compat automatically fixes this by making `torch.cuda.is_available()` return `False`, which forces libraries to detect HCCL. Make sure ascend-compat is activated before `import accelerate` or `import deepspeed`.

### DeepSpeed fails with "timer.py" errors

**Fix:**
```python
from ascend_compat.ecosystem import deepspeed_patch
deepspeed_patch.apply()
```

---

## Getting Help

1. **Run full diagnostics:** `ascend-compat doctor --full`
2. **Check error codes:** `ascend-compat error <code>`
3. **File an issue:** Include the output of `ascend-compat doctor --full` and `ascend-compat info`
