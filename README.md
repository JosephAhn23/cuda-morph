# cuda-morph

Run PyTorch workloads on non-NVIDIA hardware with minimal code changes.

<p>
  <img src="https://img.shields.io/badge/python-3.8%2B-1f6feb?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.8+" />
  <img src="https://img.shields.io/badge/status-alpha-6e40c9?style=for-the-badge" alt="Alpha status" />
  <img src="https://img.shields.io/badge/license-Apache%202.0-0d1117?style=for-the-badge" alt="Apache 2.0 license" />
  <img src="https://img.shields.io/badge/theme-dark%20first-111827?style=for-the-badge" alt="Dark first" />
</p>

![cuda-morph dark hero](assets/readme-hero-dark.svg)

**What you get:** existing CUDA-style PyTorch scripts keep running, while `cuda-morph` redirects calls to available backends (or safe CPU fallback).

---

## Why this exists

A lot of ML code assumes `torch.cuda` everywhere. On non-NVIDIA setups, that often means immediate runtime errors and expensive rewrites.

`cuda-morph` is a runtime compatibility layer that helps you keep the same workflow and ship faster.

---

## Visual walkthrough

![cuda-morph workflow dark](assets/readme-workflow-dark.svg)

1. Start with existing CUDA-oriented code.
2. Activate `cuda-morph`.
3. Run with backend-aware routing and fallback behavior.

---

## Quick start

```bash
pip install cuda-morph
```

```python
import ascend_compat
ascend_compat.activate()

# Existing CUDA-style code stays the same
model = model.cuda()
```

```bash
cuda-morph info
cuda-morph check model.py
```

---

## Core capabilities

- Zero-rewrite activation for many CUDA-style PyTorch flows.
- Backend routing across Ascend / ROCm / Intel XPU / CPU fallback paths.
- CLI tooling for environment checks, porting hints, and validation commands.
- Ecosystem patches for key libraries (current depth varies by backend).

Full matrix: [docs/compatibility_matrix.md](docs/compatibility_matrix.md)

---

## Validation snapshot

![cuda-morph cli dark](assets/readme-cli-dark.svg)

- 460+ tests passing in CPU-fallback mode.
- Architecture and shim behavior are strongly simulation-validated.
- Real hardware validation is still needed for full production confidence.

If you can test on non-NVIDIA accelerators, feedback is highly valuable: [open an issue](https://github.com/JosephAhn23/cuda-morph/issues).

---

## Backend status (current)

| Backend | Hardware | Status |
|---------|----------|--------|
| **Huawei Ascend** | 910B, 310P | Full shim + ecosystem patches (flash-attn, HuggingFace, DeepSpeed, vLLM). Needs hardware validation. |
| **AMD ROCm** | MI210, MI250X, MI300X | Detection + device routing. Ecosystem patches not yet implemented. |
| **Intel XPU** | Max 1550, Flex, Arc | Detection + device routing. Ecosystem patches not yet implemented. |
| **Cambricon** | MLU370, MLU590 | Detection + device routing. Ecosystem patches not yet implemented. |

---

## CLI

```bash
cuda-morph check model.py
cuda-morph port model.py
cuda-morph doctor
cuda-morph doctor --full
cuda-morph verify --device npu
cuda-morph bench overhead
cuda-morph info
```

---

## Development

```bash
git clone https://github.com/JosephAhn23/cuda-morph.git
cd cuda-morph
pip install -e ".[dev]"
pytest tests/ -v
pytest tests/ -v --run-hardware
```

---

## Notes on screenshots

The dark-mode visuals in this README are polished preview screenshots for presentation and storytelling. Replace with captured live outputs from your own environment when publishing benchmark claims.

---

## See also

- [STRATEGY.md](STRATEGY.md)
- [VALIDATION_STATUS.md](VALIDATION_STATUS.md)
- [MIGRATION.md](MIGRATION.md)
- [docs/README_zh.md](docs/README_zh.md)

## License

Apache 2.0
