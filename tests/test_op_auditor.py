"""Tests for the operator coverage auditor."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from ascend_compat.doctor.op_auditor import (
    AuditReport,
    OpInfo,
    _KNOWN_UNSUPPORTED_OPS,
    _KNOWN_SLOW_OPS,
    audit_model,
)


class TestAuditReport:
    """Tests for the AuditReport dataclass."""

    def test_empty_report(self) -> None:
        report = AuditReport(model_name="empty")
        assert report.coverage_pct == 100.0
        assert "All operators" in report.summary()

    def test_report_with_fallbacks(self) -> None:
        report = AuditReport(
            model_name="test",
            total_ops=10,
            native_ops=8,
            fallback_ops=2,
            ops={
                "aten::histc": OpInfo(name="aten::histc", call_count=5, status="fallback"),
                "aten::_unique2": OpInfo(name="aten::_unique2", call_count=2, status="fallback"),
            },
        )
        assert report.coverage_pct == 80.0
        summary = report.summary()
        assert "80.0%" in summary
        assert "aten::histc" in summary
        assert "aten::_unique2" in summary

    def test_report_with_slow_ops(self) -> None:
        report = AuditReport(
            model_name="test",
            total_ops=5,
            native_ops=4,
            slow_ops=1,
            ops={
                "aten::nonzero": OpInfo(
                    name="aten::nonzero", call_count=3, status="slow",
                    note="Dynamic output shape"
                ),
            },
        )
        summary = report.summary()
        assert "Slow operators" in summary
        assert "aten::nonzero" in summary


class TestKnownOps:
    """Tests for the known op lists."""

    def test_unsupported_ops_nonempty(self) -> None:
        assert len(_KNOWN_UNSUPPORTED_OPS) > 10

    def test_slow_ops_nonempty(self) -> None:
        assert len(_KNOWN_SLOW_OPS) > 0

    def test_known_ops_have_aten_prefix(self) -> None:
        for op in _KNOWN_UNSUPPORTED_OPS:
            assert "::" in op, f"Op {op} should have namespace prefix"


class TestAuditModel:
    """Tests for the audit_model function."""

    def test_audit_simple_linear(self) -> None:
        """Audit a simple linear model."""
        model = nn.Linear(10, 5)
        x = torch.randn(2, 10)
        report = audit_model(model, x, model_name="SimpleLinear")
        assert report.model_name == "SimpleLinear"
        assert report.total_ops > 0

    def test_audit_sequential(self) -> None:
        """Audit a Sequential model."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        x = torch.randn(4, 10)
        report = audit_model(model, x)
        assert report.total_ops > 0
        # Linear + ReLU + Linear should all be native
        summary = report.summary()
        assert isinstance(summary, str)

    def test_audit_conv_model(self) -> None:
        """Audit a convolutional model."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 10),
        )
        x = torch.randn(1, 3, 32, 32)
        report = audit_model(model, x, model_name="SmallCNN")
        assert report.total_ops > 0
        assert report.model_name == "SmallCNN"

    def test_audit_returns_report_type(self) -> None:
        model = nn.Linear(5, 3)
        x = torch.randn(1, 5)
        report = audit_model(model, x)
        assert isinstance(report, AuditReport)

    def test_audit_default_name(self) -> None:
        model = nn.Linear(5, 3)
        x = torch.randn(1, 5)
        report = audit_model(model, x)
        assert report.model_name == "Linear"
