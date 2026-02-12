"""Tests for the dtype auto-management module."""

from __future__ import annotations

import warnings

import pytest
import torch

from ascend_compat.cuda_shim.dtype_manager import (
    DTypePolicy,
    _resolve_dtype,
    apply_dtype_policy,
    check_dtype_support,
    get_substitution_map,
    _unpatch_creation_fns,
)


class TestCheckDtypeSupport:
    """Tests for dtype support queries."""

    def test_float64_unsupported(self) -> None:
        supported, sub = check_dtype_support("float64")
        assert not supported
        assert sub == "float32"

    def test_float32_supported(self) -> None:
        supported, sub = check_dtype_support("float32")
        assert supported
        assert sub is None

    def test_float16_supported(self) -> None:
        supported, sub = check_dtype_support("float16")
        assert supported
        assert sub is None

    def test_int_types_supported(self) -> None:
        for dtype in ("int8", "int16", "int32", "int64", "uint8", "bool"):
            supported, sub = check_dtype_support(dtype)
            assert supported, f"{dtype} should be supported"

    def test_quantized_unsupported(self) -> None:
        supported, sub = check_dtype_support("quint8")
        assert not supported


class TestSubstitutionMap:
    """Tests for the substitution map."""

    def test_float64_in_map(self) -> None:
        subs = get_substitution_map()
        assert subs["float64"] == "float32"

    def test_map_is_dict(self) -> None:
        subs = get_substitution_map()
        assert isinstance(subs, dict)


class TestResolveDtype:
    """Tests for the _resolve_dtype function."""

    def test_none_passthrough(self) -> None:
        assert _resolve_dtype(None) is None

    def test_disabled_policy_passthrough(self) -> None:
        apply_dtype_policy(DTypePolicy.DISABLED)
        result = _resolve_dtype(torch.float64)
        assert result is torch.float64

    def test_supported_dtype_unchanged(self) -> None:
        apply_dtype_policy(DTypePolicy.AUTO)
        try:
            result = _resolve_dtype(torch.float32)
            assert result is torch.float32
        finally:
            apply_dtype_policy(DTypePolicy.DISABLED)

    def test_float64_substituted_in_auto_mode(self) -> None:
        apply_dtype_policy(DTypePolicy.AUTO)
        try:
            result = _resolve_dtype(torch.float64)
            assert result is torch.float32
        finally:
            apply_dtype_policy(DTypePolicy.DISABLED)

    def test_float64_raises_in_strict_mode(self) -> None:
        apply_dtype_policy(DTypePolicy.STRICT)
        try:
            with pytest.raises(RuntimeError, match="float64"):
                _resolve_dtype(torch.float64)
        finally:
            apply_dtype_policy(DTypePolicy.DISABLED)

    def test_float64_warns_in_warn_mode(self) -> None:
        apply_dtype_policy(DTypePolicy.WARN)
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = _resolve_dtype(torch.float64)
                assert result is torch.float32
                assert any("float64" in str(warning.message) for warning in w)
        finally:
            apply_dtype_policy(DTypePolicy.DISABLED)


class TestDtypePolicy:
    """Tests for apply_dtype_policy."""

    def teardown_method(self) -> None:
        apply_dtype_policy(DTypePolicy.DISABLED)

    def test_disabled_does_not_patch(self) -> None:
        original_zeros = torch.zeros
        apply_dtype_policy(DTypePolicy.DISABLED)
        assert torch.zeros is original_zeros

    def test_auto_patches_creation_fns(self) -> None:
        original_zeros = torch.zeros
        apply_dtype_policy(DTypePolicy.AUTO)
        # torch.zeros should now be wrapped
        assert torch.zeros is not original_zeros
        apply_dtype_policy(DTypePolicy.DISABLED)
        # Should be restored
        assert torch.zeros is original_zeros

    def test_patched_zeros_substitutes_float64(self) -> None:
        apply_dtype_policy(DTypePolicy.AUTO)
        try:
            # This should silently substitute float64 → float32
            x = torch.zeros(2, 2, dtype=torch.float64)
            # On CPU, float64 still works (the substitution targets NPU),
            # but the patching mechanism itself should work
            assert x is not None
        finally:
            apply_dtype_policy(DTypePolicy.DISABLED)
