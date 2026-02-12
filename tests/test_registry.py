"""Tests for the mapping registry (cuda_shim._registry)."""

from __future__ import annotations

from ascend_compat.cuda_shim._registry import (
    Mapping,
    MappingKind,
    classify_attr,
    get_all_mappings,
    get_direct_mappings,
    get_adapted_mappings,
    get_mapping,
    get_unsupported,
)


class TestRegistry:
    """Tests for the mapping registry."""

    def test_direct_mappings_exist(self) -> None:
        """Core device/memory/stream APIs should be registered as DIRECT."""
        for attr in ("is_available", "device_count", "current_device",
                     "set_device", "synchronize", "memory_allocated",
                     "empty_cache", "manual_seed"):
            m = get_mapping(attr)
            assert m is not None, f"Missing mapping for {attr}"
            assert m.kind == MappingKind.DIRECT, f"{attr} should be DIRECT"

    def test_unsupported_mappings_exist(self) -> None:
        """Known unsupported ops should be in the registry."""
        for attr in ("memory_snapshot", "CUDAGraph"):
            m = get_mapping(attr)
            assert m is not None, f"Missing mapping for {attr}"
            assert m.kind == MappingKind.UNSUPPORTED

    def test_classify_attr_known(self) -> None:
        assert classify_attr("is_available") == "direct"
        assert classify_attr("memory_snapshot") == "unsupported"

    def test_classify_attr_unknown(self) -> None:
        assert classify_attr("some_nonexistent_thing") == "unknown"

    def test_all_mappings_is_dict(self) -> None:
        all_m = get_all_mappings()
        assert isinstance(all_m, dict)
        assert len(all_m) > 0

    def test_direct_mappings_list(self) -> None:
        directs = get_direct_mappings()
        assert all(m.kind == MappingKind.DIRECT for m in directs)
        assert len(directs) > 10  # We have at least ~20 direct mappings

    def test_unsupported_list(self) -> None:
        unsup = get_unsupported()
        assert all(m.kind == MappingKind.UNSUPPORTED for m in unsup)
        assert len(unsup) > 0

    def test_mapping_has_note(self) -> None:
        """Unsupported mappings should have guidance notes."""
        for m in get_unsupported():
            assert m.note, f"Unsupported mapping {m.cuda_name} has no note"

    def test_npu_name_matches_cuda_for_direct(self) -> None:
        """For DIRECT mappings, npu_name should equal cuda_name."""
        for m in get_direct_mappings():
            assert m.npu_name == m.cuda_name, (
                f"Direct mapping {m.cuda_name} has different npu_name: {m.npu_name}"
            )
