# cuda-morph

Run CUDA-assuming PyTorch code on Huawei Ascend NPU (and CPU as a fallback)
without changing your model code.

**Status:** Alpha, CPU-validated (450 tests passing) — not yet verified against
real Ascend/ROCm hardware. **Shim overhead:** 0.13µs/call, CPU benchmarks only.
See [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) for methodology.

## Install

```bash
pip install cuda-morph
```

## One Line

```python
import ascend_compat  # registers the "morphos" torch.compile backend

model = torch.compile(model, backend="morphos")
```

`"morphos"` audits your model's compute graph for operators known to
silently fall back to CPU (or run natively but slowly) on Ascend NPU, warns
you about them with a pointer to `cuda-morph scaffold <op>`, then delegates
actual compilation to whichever backend is right for your hardware
(`torchair` on Ascend, `inductor` on CUDA/CPU). See
[`morphos_backend.py`](src/ascend_compat/cuda_shim/morphos_backend.py).

For the zero-`torch.compile` path — patching `torch.cuda.*` calls directly —
use `ascend_compat.activate()` instead; see the package docstring for the
full four-layer architecture and all three activation modes.

## How It Works

1. `ascend_compat.activate()` intercepts `torch.cuda.*` and routes it to
   `torch.npu.*` (or ROCm/CPU) equivalents — no code changes to your model.
2. `ascend_compat.doctor` audits your model beforehand for operators that
   won't run natively, so surprises show up before deployment, not during it.
3. `ascend_compat.ecosystem` patches HuggingFace Transformers, DeepSpeed,
   flash-attn, and vLLM for the same class of CUDA-assuming bugs.
4. `ascend_compat.kernel_helper` scaffolds a real Ascend C kernel project
   for operators that need one written from scratch.

## Implementation

- **[src/ascend_compat/](src/ascend_compat/)** — the installable package
- **[bench_complete.py](bench_complete.py)** — benchmarking suite
- **[BENCHMARK_REPORT.md](BENCHMARK_REPORT.md)** — performance validation (CPU only)

## Test

```bash
pip install -e ".[dev]"
pytest tests/
```

## Business

- **TAM:** $400M-800M (enterprise GPU waste)
- **Unit Econ:** 8-12x ROI, 2-month payback
- **Value:** Save $2-4M/year per customer

## Next

1. GPU credits (AMD AI Developer Program)
2. Benchmark on MI300X
3. Compare CUDA vs ROCm
4. Customer acquisition
 
