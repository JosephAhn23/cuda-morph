"""Tests for the Ascend C kernel helper (scaffolding + spec)."""

from __future__ import annotations

import os
import tempfile

import pytest

from ascend_compat.kernel_helper.spec import (
    SUPPORTED_DTYPES,
    SUPPORTED_PATTERNS,
    OpSpec,
)
from ascend_compat.kernel_helper.scaffold import (
    _to_cann_dtype,
    _to_cpp_type,
    scaffold,
)


class TestOpSpec:
    """Test the declarative operator specification."""

    def test_valid_elementwise_spec(self) -> None:
        spec = OpSpec(
            name="MyAdd",
            inputs=[("x", "float16"), ("y", "float16")],
            outputs=[("z", "float16")],
            pattern="elementwise",
        )
        assert spec.name == "MyAdd"
        assert spec.uses_vector
        assert not spec.uses_cube
        assert len(spec.all_tensors) == 3

    def test_valid_matmul_spec(self) -> None:
        spec = OpSpec(
            name="MyMatmul",
            inputs=[("a", "float16"), ("b", "float16")],
            outputs=[("c", "float16")],
            pattern="matmul",
        )
        assert spec.uses_cube
        assert not spec.uses_vector

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            OpSpec(name="", inputs=[("x", "float16")], outputs=[("y", "float16")])

    def test_rejects_lowercase_name(self) -> None:
        with pytest.raises(ValueError, match="PascalCase"):
            OpSpec(name="myAdd", inputs=[("x", "float16")], outputs=[("y", "float16")])

    def test_rejects_unknown_pattern(self) -> None:
        with pytest.raises(ValueError, match="Unknown pattern"):
            OpSpec(
                name="MyOp",
                inputs=[("x", "float16")],
                outputs=[("y", "float16")],
                pattern="nonexistent",
            )

    def test_rejects_unsupported_dtype(self) -> None:
        with pytest.raises(ValueError, match="unsupported dtype"):
            OpSpec(
                name="MyOp",
                inputs=[("x", "float64")],
                outputs=[("y", "float16")],
            )

    def test_rejects_no_inputs(self) -> None:
        with pytest.raises(ValueError, match="at least one inputs"):
            OpSpec(name="MyOp", inputs=[], outputs=[("y", "float16")])

    def test_rejects_no_outputs(self) -> None:
        with pytest.raises(ValueError, match="at least one outputs"):
            OpSpec(name="MyOp", inputs=[("x", "float16")], outputs=[])

    def test_rejects_bad_alignment(self) -> None:
        with pytest.raises(ValueError, match="multiple of 32"):
            OpSpec(
                name="MyOp",
                inputs=[("x", "float16")],
                outputs=[("y", "float16")],
                alignment=24,
            )

    def test_to_dict(self) -> None:
        spec = OpSpec(
            name="TestOp",
            inputs=[("x", "float16")],
            outputs=[("y", "float16")],
            description="A test op",
        )
        d = spec.to_dict()
        assert d["name"] == "TestOp"
        assert d["pattern"] == "elementwise"
        assert d["uses_vector"] is True

    def test_all_supported_dtypes_accepted(self) -> None:
        for dtype in SUPPORTED_DTYPES:
            spec = OpSpec(
                name="TestOp",
                inputs=[("x", dtype)],
                outputs=[("y", dtype)],
            )
            assert spec.name == "TestOp"


class TestScaffold:
    """Test the file generation scaffolding."""

    def _make_spec(self, pattern: str = "elementwise") -> OpSpec:
        return OpSpec(
            name="FusedRMSNorm",
            inputs=[("x", "float16"), ("weight", "float16")],
            outputs=[("y", "float16")],
            pattern=pattern,
            description="Fused RMS Normalization for LLM inference",
        )

    def test_generates_all_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = self._make_spec()
            files = scaffold(spec, tmpdir)

            assert "CMakeLists.txt" in files
            assert "op_host/fusedrmsnorm_tiling.h" in files
            assert "op_host/fusedrmsnorm.cpp" in files
            assert "op_kernel/fusedrmsnorm.cpp" in files
            assert "README.md" in files

            # All files should be written to disk
            for relpath in files:
                fullpath = os.path.join(tmpdir, relpath)
                assert os.path.isfile(fullpath), f"Missing: {relpath}"

    def test_tiling_header_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = self._make_spec()
            files = scaffold(spec, tmpdir)

            header = files["op_host/fusedrmsnorm_tiling.h"]
            assert "FUSEDRMSNORM_TILING_H" in header  # include guard
            assert "totalLength" in header
            assert "tileNum" in header
            assert "tileLength" in header
            assert "REGISTER_TILING_DATA_CLASS" in header

    def test_host_impl_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = self._make_spec()
            files = scaffold(spec, tmpdir)

            host = files["op_host/fusedrmsnorm.cpp"]
            assert "REG_OP(FusedRMSNorm)" in host
            assert "TilingFunc" in host
            assert "DT_FLOAT16" in host
            assert "32-byte alignment" in host or "alignment" in host

    def test_kernel_impl_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = self._make_spec()
            files = scaffold(spec, tmpdir)

            kernel = files["op_kernel/fusedrmsnorm.cpp"]
            assert "KernelFusedRMSNorm" in kernel
            assert "CopyIn" in kernel
            assert "Compute" in kernel
            assert "CopyOut" in kernel
            assert "__global__ __aicore__" in kernel
            assert "Pipeline" in kernel or "pipe_" in kernel

    def test_matmul_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = self._make_spec(pattern="matmul")
            files = scaffold(spec, tmpdir)

            kernel = files["op_kernel/fusedrmsnorm.cpp"]
            assert "Cube" in kernel  # Should reference Cube unit
            assert "16×16" in kernel or "16x16" in kernel

    def test_reduction_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = self._make_spec(pattern="reduction")
            files = scaffold(spec, tmpdir)

            kernel = files["op_kernel/fusedrmsnorm.cpp"]
            assert "ReduceSum" in kernel or "Reduce" in kernel

    def test_cmake_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = self._make_spec()
            files = scaffold(spec, tmpdir)

            cmake = files["CMakeLists.txt"]
            assert "ASCEND_HOME" in cmake
            assert "cmake_minimum_required" in cmake
            assert "CXX" in cmake

    def test_readme_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = self._make_spec()
            files = scaffold(spec, tmpdir)

            readme = files["README.md"]
            assert "FusedRMSNorm" in readme
            assert "CopyIn" in readme
            assert "Pipeline" in readme or "pipeline" in readme
            assert "double-buffer" in readme.lower() or "ping-pong" in readme.lower()


class TestTypeMappings:
    """Test C++ and CANN dtype mappings."""

    def test_float16_to_half(self) -> None:
        assert _to_cpp_type("float16") == "half"

    def test_float32_to_float(self) -> None:
        assert _to_cpp_type("float32") == "float"

    def test_bfloat16_mapping(self) -> None:
        assert _to_cpp_type("bfloat16") == "bfloat16_t"

    def test_int8_mapping(self) -> None:
        assert _to_cpp_type("int8") == "int8_t"

    def test_cann_float16(self) -> None:
        assert _to_cann_dtype("float16") == "FLOAT16"

    def test_cann_float32(self) -> None:
        assert _to_cann_dtype("float32") == "FLOAT"

    def test_cann_bf16(self) -> None:
        assert _to_cann_dtype("bfloat16") == "BF16"
