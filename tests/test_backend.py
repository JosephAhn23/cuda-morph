"""Tests for the backend detection module (_backend)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ascend_compat._backend import (
    Backend,
    detect_backends,
    has_cuda,
    has_npu,
    preferred_backend,
    translate_device_string,
)


class TestBackendEnum:
    """Tests for the Backend enum."""

    def test_values(self) -> None:
        assert Backend.NPU.value == "npu"
        assert Backend.CUDA.value == "cuda"
        assert Backend.CPU.value == "cpu"


class TestDetection:
    """Tests for backend detection (uses conftest fixtures)."""

    def test_cpu_only(self, cpu_only_backend: None) -> None:
        assert preferred_backend() == Backend.CPU
        assert not has_npu()

    def test_cuda_detected(self, cuda_backend: None) -> None:
        assert preferred_backend() == Backend.CUDA

    def test_npu_detected(self, npu_backend: None) -> None:
        assert preferred_backend() == Backend.NPU
        assert has_npu()


class TestTranslateDeviceString:
    """Tests for device string translation."""

    def test_cuda_to_npu(self, npu_backend: None) -> None:
        assert translate_device_string("cuda") == "npu"
        assert translate_device_string("cuda:0") == "npu:0"
        assert translate_device_string("cuda:3") == "npu:3"

    def test_cpu_unchanged(self, npu_backend: None) -> None:
        assert translate_device_string("cpu") == "cpu"

    def test_npu_unchanged(self, npu_backend: None) -> None:
        assert translate_device_string("npu:0") == "npu:0"

    def test_cuda_on_cuda_system(self, cuda_backend: None) -> None:
        """On a CUDA system, cuda strings should pass through."""
        assert translate_device_string("cuda") == "cuda"
        assert translate_device_string("cuda:0") == "cuda:0"

    def test_cuda_on_cpu_system(self, cpu_only_backend: None) -> None:
        """On CPU-only, cuda strings should become cpu."""
        assert translate_device_string("cuda") == "cpu"
