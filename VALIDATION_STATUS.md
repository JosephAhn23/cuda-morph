# Validation Status

Last updated: 2026-02-11

## Summary

**ascend-compat is simulation-validated, not hardware-validated.**

The architecture, test suite, and patching machinery work correctly in
CPU-fallback mode. The CUDA-to-NPU argument mappings are based on Huawei's
documentation, not empirical NPU execution.

---

## What IS validated

| Component | How validated | Confidence |
|-----------|--------------|------------|
| `torch.cuda.is_available()` → `False` | Integration test on real `torch` | **High** |
| `torch.device("cuda")` → `torch.device("npu")` | Integration test | **High** |
| `Tensor.cuda()` / `Module.cuda()` → `.npu()` | Integration test | **High** |
| PatchManager reference counting | Unit + integration test | **High** |
| PatchManager atomic activation/rollback | Unit test | **High** |
| Thread-safe activation/deactivation | Concurrent thread test | **High** |
| ShapeBucketer LRU cache + eviction | Unit + stress test (1024+ entries) | **High** |
| CLI commands (`check`, `port`, `info`, `doctor`, `error`) | End-to-end smoke test | **High** |
| Deprecation warning for v0.2.x users | Unit test | **High** |
| CANN error code translation | Unit test (50+ codes) | **High** |
| CPU fallback detection + monitoring | Unit test | **High** |
| Quantization compatibility database | Unit test | **Medium** |
| Ecosystem patch version guards | Unit test | **Medium** |
| Ecosystem patch-landing verification | Unit test (mock environment) | **Medium** |

## What is NOT validated

| Component | What's missing | Risk |
|-----------|---------------|------|
| `flash_attn_func` argument mapping | Never run on NPU; `dropout_p→keep_prob`, `causal→next_tockens` untested | **Critical** |
| `flash_attn_varlen_func` padding logic | Pad/unpad may introduce numerical drift | **High** |
| `flash_attn_with_kvcache` cache update | In-place KV-cache update untested on NPU memory model | **High** |
| HuggingFace `transformers_patch` | Patches may break on newer transformers versions | **High** |
| DeepSpeed HCCL backend selection | `init_distributed` patch untested in real distributed setting | **High** |
| vLLM attention backend routing | Never run with vllm-ascend | **High** |
| `npu_fusion_attention` numerical accuracy | No comparison against CUDA flash_attn output | **Critical** |
| Dtype auto-substitution correctness | float64→float32 may change model behavior | **Medium** |
| `kernel_helper` generated code | Generated Ascend C code never compiled | **Medium** |
| `_KNOWN_HASHES` security check | Empty — always reports "unknown" | **Low** (honest) |
| Memory bandwidth benchmarks on NPU | Only tested on CPU | **Low** |

## What would change the status

### Minimum viable hardware validation (estimated: 2 weeks with NPU access)

1. Run `pytest tests/test_hardware.py --run-hardware` on Ascend 910B
2. Fix whatever breaks (expect 20-50 argument mapping issues)
3. Run `ascend-compat verify --device npu` and commit the report
4. Run `examples/huggingface_inference.py` on NPU with GPT-2 and BERT-tiny
5. Run flash_attn numerical comparison: `OperatorVerifier.verify_flash_attention()`

### Full validation (estimated: 1-2 months)

6. Run Llama-7B inference end-to-end on NPU
7. Run DeepSpeed ZeRO-2 distributed training (2+ NPUs)
8. Produce benchmark artifacts: tokens/sec, memory profile, latency percentiles
9. Populate `_KNOWN_HASHES` from official torch_npu releases
10. Compile at least one `kernel_helper`-generated kernel

## How to get hardware access

- **Huawei Cloud ModelArts**: Ascend 910B instances available (not free tier)
- **Ascend Community Seed Fund**: Hardware credits for open-source projects
- **Academic programs**: Huawei partners with universities (CANARIE, direct)
- **Community contribution**: If you have NPU access, run the hardware tests and submit results

## Operator verification

ascend-compat includes an `OperatorVerifier` that empirically compares shimmed
operator output against CPU reference implementations:

```bash
# On CPU (simulation — useful for catching API errors, not numerical issues):
ascend-compat verify --device cpu

# On real NPU (the actual validation):
ascend-compat verify --device npu
```

The verifier checks:
- `matmul` — basic GEMM correctness
- `softmax` — numerical stability
- `LayerNorm` — compound operation correctness
- `flash_attn_func` — the critical shimmed operator
- `flash_attn_func(causal=True)` — causal masking

Each check reports max absolute error, mean absolute error, and cosine
similarity against the float32 CPU reference.

## Test suite breakdown

| Category | Count | Requires hardware |
|----------|-------|-------------------|
| Unit tests | ~250 | No |
| Integration tests | ~80 | No |
| Stress tests | ~20 | No |
| Performance benchmarks | ~10 | No |
| End-to-end smoke tests | ~30 | No |
| **Hardware tests** | **~15** | **Yes** |
| **Total** | **~405** | |

To run everything except hardware tests:
```bash
pytest tests/ -v
```

To run hardware tests (on Ascend NPU):
```bash
pytest tests/ -v --run-hardware
```
