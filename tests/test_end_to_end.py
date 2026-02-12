"""End-to-end smoke tests — full stack, no mocks, CPU fallback.

These tests prove the entire codebase works together on a real Python
interpreter with real torch objects.  They exercise the code paths that
users actually hit, not mock-isolated units.

Unlike test_integration.py (which tests individual patches), these tests
simulate actual user workflows:
  - Import ascend_compat → activate → use torch → check telemetry → deactivate
  - Run CLI commands against real torch installation
  - Exercise ecosystem patches without their target libraries installed
  - Verify the examples would at least import without crashing

No mocks.  No fakes.  If it fails here, it fails for users.
"""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest
import torch


class TestUserWorkflow:
    """Simulate what a real user does."""

    def test_minimal_workflow(self):
        """The smallest possible ascend-compat usage."""
        import ascend_compat

        ascend_compat.activate()
        assert ascend_compat.is_activated()

        # User code: create tensor, check device
        t = torch.zeros(4, 4)
        assert t.shape == (4, 4)

        # This is what real code does:
        result = torch.cuda.is_available()
        assert isinstance(result, bool)

        ascend_compat.deactivate()
        assert not ascend_compat.is_activated()

    def test_device_creation_after_activation(self):
        """torch.device("cpu") should always work, activated or not."""
        from ascend_compat.cuda_shim import activate, deactivate

        activate()
        d = torch.device("cpu")
        assert str(d) == "cpu"
        deactivate()

        d2 = torch.device("cpu")
        assert str(d2) == "cpu"

    def test_tensor_ops_unaffected(self):
        """Tensor operations should work identically with shim active."""
        from ascend_compat.cuda_shim import activate, deactivate

        a = torch.randn(3, 3)
        b = torch.randn(3, 3)
        expected = torch.matmul(a, b)

        activate()
        result = torch.matmul(a, b)
        deactivate()

        assert torch.allclose(expected, result)


class TestCLISmoke:
    """Verify CLI commands don't crash against real torch."""

    def test_info_command(self):
        """ascend-compat info should produce output."""
        from ascend_compat.cli import show_info

        info = show_info()
        assert "PyTorch version" in info
        assert "ascend-compat" in info

    def test_check_command_on_self(self):
        """ascend-compat check should work on a real Python file."""
        from ascend_compat.cli import check_file
        import ascend_compat

        # Check one of our own example files or source files
        init_path = ascend_compat.__file__
        if init_path:
            report = check_file(init_path)
            assert report.total_cuda_refs >= 0  # Might have zero refs

    def test_compile_info(self):
        """ascend-compat compile info should return valid data."""
        from ascend_compat.cuda_shim.compile_helpers import get_compile_info

        info = get_compile_info()
        assert "recommended_backend" in info
        assert isinstance(info["recommended_backend"], str)

    def test_doctor_version_check(self):
        """ascend-compat doctor should not crash."""
        from ascend_compat.doctor.version_check import check_versions

        results = check_versions()
        assert isinstance(results, list)

    def test_error_code_translation(self):
        """Error code translation should work."""
        from ascend_compat.doctor.error_codes import translate_error

        info = translate_error("507035")
        assert info is not None


class TestEcosystemPatchSafety:
    """Verify ecosystem patches don't crash when libraries are missing."""

    def test_transformers_patch_safe_without_transformers(self):
        """transformers_patch.apply() should not crash if transformers is missing."""
        from ascend_compat.ecosystem import transformers_patch
        # Reset state
        transformers_patch._applied = False
        # This should be a no-op (no NPU, or no transformers)
        transformers_patch.apply()

    def test_deepspeed_patch_safe_without_deepspeed(self):
        """deepspeed_patch.apply() should not crash if deepspeed is missing."""
        from ascend_compat.ecosystem import deepspeed_patch
        deepspeed_patch._applied = False
        deepspeed_patch.apply()

    def test_vllm_patch_safe_without_vllm(self):
        """vllm_patch.apply() should not crash if vllm is missing."""
        from ascend_compat.ecosystem import vllm_patch
        vllm_patch._applied = False
        vllm_patch.apply()

    def test_flash_attn_shim_importable(self):
        """The flash_attn shim should import without torch_npu."""
        from ascend_compat.ecosystem.flash_attn import flash_attn_func
        assert callable(flash_attn_func)


class TestSecurityCheckHonesty:
    """Verify security check is honest about what it can and can't verify."""

    def test_returns_unknown_not_ok_without_hashes(self):
        """Without known hashes, status should be 'unknown' not 'ok'."""
        from ascend_compat.doctor.security_check import verify_torch_npu_integrity

        result = verify_torch_npu_integrity()
        # torch_npu is not installed in test env, so expect "warning"
        # If it WERE installed but hashes are empty, it would be "unknown"
        assert result.status in ("warning", "unknown")
        assert result.status != "ok"  # Can't be "ok" without verified hashes

    def test_cann_check_without_ascend_home(self):
        """CANN library check should warn if ASCEND_HOME_PATH not set."""
        import os
        from ascend_compat.doctor.security_check import verify_cann_libraries

        old = os.environ.pop("ASCEND_HOME_PATH", None)
        try:
            results = verify_cann_libraries()
            assert len(results) >= 1
            assert results[0].status == "warning"
        finally:
            if old is not None:
                os.environ["ASCEND_HOME_PATH"] = old

    def test_full_security_check_runs(self):
        """Full security check should complete without crashing."""
        from ascend_compat.doctor.security_check import full_security_check, format_security_report

        results = full_security_check()
        report = format_security_report(results)
        assert "security check" in report.lower()
        # Should NOT claim things are OK when we can't verify
        assert "[OK]" not in report or "CANN" not in report


class TestBenchmarkEndToEnd:
    """Verify benchmarks produce real results on CPU."""

    def test_shim_overhead_bench(self):
        from ascend_compat.bench import ShimOverheadBench

        report = ShimOverheadBench(iterations=100).run()
        assert len(report.results) > 0
        for r in report.results:
            assert r.per_call_us > 0
            assert r.iterations == 100

    def test_op_latency_bench_cpu(self):
        from ascend_compat.bench import OpLatencyBench

        report = OpLatencyBench(device="cpu", iterations=10).run()
        assert len(report.results) > 0

    def test_system_fingerprint_complete(self):
        from ascend_compat.bench import get_system_fingerprint

        fp = get_system_fingerprint()
        assert fp["torch_version"] == torch.__version__
        assert fp["ascend_compat_version"] is not None
        assert fp["python_version"] is not None

    def test_csv_export_parseable(self):
        """CSV export should be parseable by csv.reader."""
        import csv
        import io
        from ascend_compat.bench import ShimOverheadBench

        report = ShimOverheadBench(iterations=50).run()
        csv_text = report.to_csv()

        # Should parse without errors
        reader = csv.reader(io.StringIO(csv_text))
        rows = [r for r in reader if not r[0].startswith("#")]
        assert len(rows) >= 2  # header + at least 1 data row
        assert rows[0][0] == "operation"


class TestShapeBucketerEndToEnd:
    """Verify ShapeBucketer works on real tensors."""

    def test_pad_real_tensor(self):
        from ascend_compat.cuda_shim.compile_helpers import ShapeBucketer

        bucketer = ShapeBucketer(buckets=[32, 64, 128])
        t = torch.randn(10, 50)  # 50 doesn't match any bucket

        padded = bucketer.pad(t, dim=1)
        assert padded.shape == (10, 64)  # Rounded up to 64
        # Original data preserved
        assert torch.equal(padded[:, :50], t)
        # Padding is zero
        assert torch.all(padded[:, 50:] == 0)

    def test_bucketer_stats_accurate(self):
        from ascend_compat.cuda_shim.compile_helpers import ShapeBucketer

        bucketer = ShapeBucketer(buckets=[32, 64, 128])

        for _ in range(10):
            t = torch.randn(5, 20)
            bucketer.pad(t, dim=1)

        stats = bucketer.stats()
        assert stats["total_calls"] == 10
        assert 32 in stats["bucket_hits"]  # 20 rounds up to 32

    def test_safe_compile_fallback(self):
        """safe_compile should return the model unchanged if compile fails."""
        from ascend_compat.cuda_shim.compile_helpers import safe_compile

        model = torch.nn.Linear(10, 5)

        # This should work (either compile or graceful fallback)
        result = safe_compile(model, backend="eager")
        # Result should be callable
        output = result(torch.randn(3, 10))
        assert output.shape == (3, 5)


class TestExamplesImportable:
    """Verify example scripts at least parse without syntax errors."""

    def test_examples_are_valid_python(self):
        """All example scripts should be valid Python syntax."""
        import ast
        from pathlib import Path

        examples_dir = Path(__file__).parent.parent / "examples"
        if not examples_dir.exists():
            pytest.skip("examples/ directory not found")

        for py_file in examples_dir.glob("*.py"):
            source = py_file.read_text(encoding="utf-8")
            try:
                ast.parse(source, filename=str(py_file))
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {py_file.name}: {e}")
