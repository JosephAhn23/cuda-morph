"""Tests for the pluggable backend registry and multi-vendor support."""

from __future__ import annotations

import pytest

from ascend_compat.backends import BACKEND_REGISTRY, BackendInfo
from ascend_compat.backends.ascend import AscendBackend
from ascend_compat.backends.cambricon import CambriconBackend
from ascend_compat._backend import Backend


class TestBackendRegistry:
    """Test the backend registry structure."""

    def test_registry_is_dict(self):
        assert isinstance(BACKEND_REGISTRY, dict)

    def test_registry_has_ascend(self):
        assert "ascend" in BACKEND_REGISTRY

    def test_registry_has_cambricon(self):
        assert "cambricon" in BACKEND_REGISTRY

    def test_all_backends_subclass_info(self):
        for name, cls in BACKEND_REGISTRY.items():
            assert issubclass(cls, BackendInfo), f"{name} must subclass BackendInfo"

    def test_all_backends_have_required_attrs(self):
        required = ["name", "device_type", "adapter_module",
                     "collective_backend", "visible_devices_env"]
        for name, cls in BACKEND_REGISTRY.items():
            for attr in required:
                val = getattr(cls, attr)
                assert val, f"{name}.{attr} must not be empty"


class TestAscendBackend:
    """Test Ascend backend configuration."""

    def test_name(self):
        assert AscendBackend.name == "ascend"

    def test_device_type(self):
        assert AscendBackend.device_type == "npu"

    def test_adapter_module(self):
        assert AscendBackend.adapter_module == "torch_npu"

    def test_collective(self):
        assert AscendBackend.collective_backend == "hccl"

    def test_visible_devices(self):
        assert AscendBackend.visible_devices_env == "ASCEND_RT_VISIBLE_DEVICES"

    def test_display_name(self):
        assert "Ascend" in AscendBackend.display_name

    def test_is_available_returns_bool(self):
        # On CI without NPU, this should return False
        result = AscendBackend.is_available()
        assert isinstance(result, bool)

    def test_device_count_returns_int(self):
        result = AscendBackend.device_count()
        assert isinstance(result, int)
        assert result >= 0

    def test_summary_returns_dict(self):
        summary = AscendBackend.summary()
        assert isinstance(summary, dict)
        assert summary["name"] == "ascend"
        assert summary["device_type"] == "npu"
        assert "available" in summary


class TestCambriconBackend:
    """Test Cambricon backend configuration."""

    def test_name(self):
        assert CambriconBackend.name == "cambricon"

    def test_device_type(self):
        assert CambriconBackend.device_type == "mlu"

    def test_adapter_module(self):
        assert CambriconBackend.adapter_module == "torch_mlu"

    def test_collective(self):
        assert CambriconBackend.collective_backend == "cncl"

    def test_visible_devices(self):
        assert CambriconBackend.visible_devices_env == "MLU_VISIBLE_DEVICES"

    def test_display_name(self):
        assert "Cambricon" in CambriconBackend.display_name

    def test_docs_url(self):
        assert "Cambricon" in CambriconBackend.docs_url

    def test_is_available_returns_false_without_hardware(self):
        # On CI without MLU, this should return False (not crash)
        result = CambriconBackend.is_available()
        assert result is False

    def test_device_count_returns_zero_without_hardware(self):
        result = CambriconBackend.device_count()
        assert result == 0

    def test_get_device_name_without_hardware(self):
        name = CambriconBackend.get_device_name()
        assert isinstance(name, str)
        assert len(name) > 0

    def test_adapter_version_none_without_install(self):
        version = CambriconBackend.get_adapter_version()
        # torch_mlu not installed → None
        assert version is None

    def test_summary_returns_dict(self):
        summary = CambriconBackend.summary()
        assert isinstance(summary, dict)
        assert summary["name"] == "cambricon"
        assert summary["device_type"] == "mlu"
        assert summary["available"] is False


class TestBackendEnum:
    """Test Backend enum includes MLU."""

    def test_mlu_exists(self):
        assert Backend.MLU.value == "mlu"

    def test_all_values(self):
        values = {b.value for b in Backend}
        assert "npu" in values
        assert "mlu" in values
        assert "cuda" in values
        assert "cpu" in values


class TestHasMlu:
    """Test the has_mlu() predicate."""

    def test_has_mlu_returns_bool(self):
        import ascend_compat
        result = ascend_compat.has_mlu()
        assert isinstance(result, bool)

    def test_has_mlu_false_without_hardware(self):
        import ascend_compat
        # No MLU hardware in CI
        assert ascend_compat.has_mlu() is False
