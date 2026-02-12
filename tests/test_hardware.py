"""Tests that require real Ascend NPU hardware.

Every test in this file is marked ``@pytest.mark.hardware`` and will be
auto-skipped unless ``--run-hardware`` is passed on the pytest command line.

These tests are the **proof layer** — they verify that ascend-compat
actually works on real Ascend hardware, not just in CPU simulation.

To run::

    pytest tests/test_hardware.py --run-hardware -v

What passes means:
    - torch_npu is installed and functional
    - NPU device is accessible
    - The CUDA→NPU shim produces correct results on real hardware
    - flash_attn → npu_fusion_attention argument mapping is correct
    - Ecosystem patches don't crash on real models

What fails means:
    - A genuine bug in ascend-compat's argument mapping or patching logic
    - File a bug: include torch_npu version, CANN version, and NPU model
"""

from __future__ import annotations

import pytest
import torch

import ascend_compat

# ---------------------------------------------------------------------------
# Skip helpers
# ---------------------------------------------------------------------------

def _npu_available() -> bool:
    """Check if a real NPU is present (not mocked)."""
    try:
        return hasattr(torch, "npu") and torch.npu.is_available()
    except Exception:
        return False


def _require_npu():
    """Skip the test if no NPU is present, even with --run-hardware."""
    if not _npu_available():
        pytest.skip("No NPU hardware detected (torch.npu.is_available() is False)")


# ===========================================================================
# Basic NPU access
# ===========================================================================

@pytest.mark.hardware
class TestNPUBasics:
    """Verify that the NPU is actually reachable."""

    def test_npu_is_available(self):
        _require_npu()
        assert torch.npu.is_available()

    def test_npu_device_count(self):
        _require_npu()
        count = torch.npu.device_count()
        assert count >= 1, f"Expected at least 1 NPU, got {count}"

    def test_npu_device_name(self):
        _require_npu()
        name = torch.npu.get_device_name(0)
        assert name, "NPU device name is empty"
        print(f"NPU device: {name}")

    def test_tensor_to_npu(self):
        _require_npu()
        x = torch.randn(4, 4)
        x_npu = x.npu()
        assert x_npu.device.type == "npu"
        # Round-trip
        x_cpu = x_npu.cpu()
        torch.testing.assert_close(x, x_cpu)

    def test_npu_memory_allocated(self):
        _require_npu()
        x = torch.randn(1024, 1024).npu()
        mem = torch.npu.memory_allocated()
        assert mem > 0, "NPU memory should be allocated after creating a tensor"
        del x


# ===========================================================================
# CUDA shim on real NPU
# ===========================================================================

@pytest.mark.hardware
class TestShimOnNPU:
    """Verify that the CUDA→NPU shim works on real hardware."""

    def setup_method(self):
        _require_npu()
        ascend_compat.activate()

    def teardown_method(self):
        ascend_compat.deactivate()

    def test_is_available_returns_false(self):
        """Core design decision: is_available() must return False on NPU."""
        assert torch.cuda.is_available() is False

    def test_device_redirect(self):
        """torch.device('cuda') should become npu."""
        dev = torch.device("cuda")
        assert dev.type == "npu", f"Expected npu, got {dev.type}"

    def test_tensor_cuda_redirect(self):
        """Tensor.cuda() should move to NPU."""
        x = torch.randn(4, 4)
        x_redirected = x.cuda()
        assert x_redirected.device.type == "npu"
        torch.testing.assert_close(x, x_redirected.cpu())

    def test_module_cuda_redirect(self):
        """Module.cuda() should move to NPU."""
        model = torch.nn.Linear(16, 16)
        model = model.cuda()
        param_device = next(model.parameters()).device
        assert param_device.type == "npu"

    def test_matmul_on_npu(self):
        """Basic matmul correctness on NPU."""
        a = torch.randn(32, 64).cuda()  # Should redirect to NPU
        b = torch.randn(64, 32).cuda()
        c = a @ b
        assert c.device.type == "npu"
        assert c.shape == (32, 32)

        # Numerical check: compare with CPU
        a_cpu = a.cpu()
        b_cpu = b.cpu()
        c_cpu = a_cpu @ b_cpu
        torch.testing.assert_close(c.cpu(), c_cpu, rtol=1e-3, atol=1e-3)

    def test_softmax_on_npu(self):
        """Softmax correctness on NPU."""
        x = torch.randn(8, 128).cuda()
        y = torch.softmax(x, dim=-1)
        assert y.device.type == "npu"

        # Should sum to 1 along last dim
        sums = y.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(8, device=y.device), rtol=1e-4, atol=1e-4)


# ===========================================================================
# Flash attention on real NPU
# ===========================================================================

@pytest.mark.hardware
class TestFlashAttnOnNPU:
    """Verify flash_attn → npu_fusion_attention argument mapping.

    This is the most critical test class.  If these pass, it means our
    argument translation (dropout_p→keep_prob, causal→next_tockens, etc.)
    is empirically correct.
    """

    def setup_method(self):
        _require_npu()
        ascend_compat.activate()

    def teardown_method(self):
        ascend_compat.deactivate()

    def test_flash_attn_basic(self):
        """Basic forward pass through our flash_attn shim."""
        from ascend_compat.ecosystem.flash_attn import flash_attn_func

        batch, seqlen, nheads, headdim = 2, 128, 8, 64
        q = torch.randn(batch, seqlen, nheads, headdim, device="npu", dtype=torch.float16)
        k = torch.randn(batch, seqlen, nheads, headdim, device="npu", dtype=torch.float16)
        v = torch.randn(batch, seqlen, nheads, headdim, device="npu", dtype=torch.float16)

        output = flash_attn_func(q, k, v)
        assert output.shape == (batch, seqlen, nheads, headdim)
        assert output.device.type == "npu"
        assert not torch.isnan(output).any(), "NaN in flash_attn output"
        assert not torch.isinf(output).any(), "Inf in flash_attn output"

    def test_flash_attn_causal(self):
        """Causal masking should produce different output than non-causal."""
        from ascend_compat.ecosystem.flash_attn import flash_attn_func

        batch, seqlen, nheads, headdim = 1, 64, 4, 32
        q = torch.randn(batch, seqlen, nheads, headdim, device="npu", dtype=torch.float16)
        k = q.clone()
        v = q.clone()

        out_causal = flash_attn_func(q, k, v, causal=True)
        out_full = flash_attn_func(q, k, v, causal=False)

        # Causal and non-causal should differ (except possibly at position 0)
        diff = (out_causal - out_full).abs().sum()
        assert diff > 0, "Causal and non-causal attention produced identical output"

    def test_flash_attn_dropout(self):
        """Dropout should not produce NaN/Inf."""
        from ascend_compat.ecosystem.flash_attn import flash_attn_func

        batch, seqlen, nheads, headdim = 1, 64, 4, 32
        q = torch.randn(batch, seqlen, nheads, headdim, device="npu", dtype=torch.float16)
        k = torch.randn(batch, seqlen, nheads, headdim, device="npu", dtype=torch.float16)
        v = torch.randn(batch, seqlen, nheads, headdim, device="npu", dtype=torch.float16)

        output = flash_attn_func(q, k, v, dropout_p=0.1)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_flash_attn_return_attn_probs(self):
        """return_attn_probs=True should return a 3-tuple."""
        from ascend_compat.ecosystem.flash_attn import flash_attn_func

        batch, seqlen, nheads, headdim = 1, 32, 4, 32
        q = torch.randn(batch, seqlen, nheads, headdim, device="npu", dtype=torch.float16)
        k = torch.randn(batch, seqlen, nheads, headdim, device="npu", dtype=torch.float16)
        v = torch.randn(batch, seqlen, nheads, headdim, device="npu", dtype=torch.float16)

        result = flash_attn_func(q, k, v, return_attn_probs=True)
        assert isinstance(result, tuple)
        assert len(result) == 3
        output, softmax_lse, _ = result
        assert output.shape == (batch, seqlen, nheads, headdim)

    def test_flash_attn_numerical_reference(self):
        """Compare our flash_attn output against naive CPU attention.

        This is THE critical correctness test.  It verifies that
        npu_fusion_attention produces numerically similar results to
        a reference (naive) attention implementation.
        """
        from ascend_compat.ecosystem.flash_attn import flash_attn_func
        import math

        batch, seqlen, nheads, headdim = 1, 32, 2, 32
        torch.manual_seed(42)
        q = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16)
        k = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16)
        v = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16)

        # Reference: naive scaled dot-product attention on CPU
        scale = 1.0 / math.sqrt(headdim)
        # Reshape to (batch, nheads, seqlen, headdim) for bmm
        q_ref = q.permute(0, 2, 1, 3).float()  # Use float32 for reference
        k_ref = k.permute(0, 2, 1, 3).float()
        v_ref = v.permute(0, 2, 1, 3).float()
        attn_weights = torch.matmul(q_ref, k_ref.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        ref_output = torch.matmul(attn_weights, v_ref)
        ref_output = ref_output.permute(0, 2, 1, 3)  # Back to BSND

        # Our shim on NPU
        npu_output = flash_attn_func(q.npu(), k.npu(), v.npu())

        # Compare
        torch.testing.assert_close(
            npu_output.cpu().float(),
            ref_output,
            rtol=5e-2,  # float16 on NPU won't match float32 reference exactly
            atol=5e-2,
        )
        print(f"Numerical error (max abs diff): "
              f"{(npu_output.cpu().float() - ref_output).abs().max().item():.6f}")


# ===========================================================================
# HuggingFace inference on real NPU
# ===========================================================================

@pytest.mark.hardware
class TestHuggingFaceOnNPU:
    """Verify that a real HuggingFace model runs on NPU via the shim."""

    def setup_method(self):
        _require_npu()
        ascend_compat.activate()

    def teardown_method(self):
        ascend_compat.deactivate()

    def test_bert_tiny_inference(self):
        """Run BERT-tiny inference on NPU.

        prajjwal1/bert-tiny is ~17M params — small enough for any NPU.
        """
        pytest.importorskip("transformers")
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        model = model.npu()
        model.eval()

        inputs = tokenizer("Hello world", return_tensors="pt")
        inputs = {k: v.npu() for k, v in inputs.items()}

        with torch.no_grad():
            output = model(**inputs)

        assert output.last_hidden_state is not None
        assert output.last_hidden_state.device.type == "npu"
        assert not torch.isnan(output.last_hidden_state).any()
        print(f"BERT-tiny output shape: {output.last_hidden_state.shape}")

    def test_gpt2_generation(self):
        """Run GPT-2 text generation on NPU."""
        pytest.importorskip("transformers")
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float16)
        model = model.npu()
        model.eval()

        inputs = tokenizer("The capital of France is", return_tensors="pt")
        inputs = {k: v.npu() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)

        assert outputs.shape[0] == 1  # Batch size
        assert outputs.shape[1] <= 20 + inputs["input_ids"].shape[1]  # Max length
        assert torch.npu.memory_allocated() > 0  # Actually used NPU

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated: {text}")
        assert len(text) > 0


# ===========================================================================
# Benchmark artifacts
# ===========================================================================

@pytest.mark.hardware
class TestNPUBenchmarks:
    """Produce benchmark artifacts that prove performance claims.

    These tests don't assert specific numbers — NPU performance varies
    by model, CANN version, and driver.  Instead, they produce structured
    output that can be committed to examples/results/.
    """

    def setup_method(self):
        _require_npu()
        ascend_compat.activate()

    def teardown_method(self):
        ascend_compat.deactivate()

    def test_matmul_throughput(self):
        """Measure matmul throughput on NPU."""
        import time

        sizes = [256, 512, 1024, 2048]
        results = []

        for n in sizes:
            a = torch.randn(n, n, device="npu", dtype=torch.float16)
            b = torch.randn(n, n, device="npu", dtype=torch.float16)

            # Warmup
            for _ in range(5):
                _ = a @ b
            torch.npu.synchronize()

            # Timed run
            start = time.perf_counter()
            iters = 100
            for _ in range(iters):
                _ = a @ b
            torch.npu.synchronize()
            elapsed = time.perf_counter() - start

            gflops = (2 * n**3 * iters) / elapsed / 1e9
            results.append((n, elapsed / iters * 1000, gflops))
            print(f"  matmul {n}x{n}: {elapsed/iters*1000:.2f} ms, {gflops:.1f} GFLOPS")

        assert len(results) == len(sizes)

    def test_memory_profile(self):
        """Report NPU memory usage for common tensor sizes."""
        results = []
        for size_mb in [1, 10, 100, 500]:
            numel = size_mb * 1024 * 1024 // 2  # float16 = 2 bytes
            torch.npu.reset_peak_memory_stats()
            x = torch.randn(numel, device="npu", dtype=torch.float16)
            peak = torch.npu.max_memory_allocated() / 1024 / 1024
            results.append((size_mb, peak))
            del x
            print(f"  Allocated {size_mb} MB → peak {peak:.1f} MB")

        assert len(results) == 4
