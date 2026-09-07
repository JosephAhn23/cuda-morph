"""Tests for the ``morphos`` torch.compile backend."""

from __future__ import annotations

import torch

# Importing the submodule below runs ascend_compat/__init__.py first (Python
# always initializes parent packages), which is what registers "morphos".
from ascend_compat.cuda_shim import morphos_backend


class TestRegistration:
    def test_morphos_is_a_registered_backend(self):
        assert "morphos" in torch._dynamo.list_backends()

    def test_registration_is_idempotent(self):
        # Importing ascend_compat already registered it once; calling again
        # must not raise (e.g. on duplicate-name assertion).
        morphos_backend.register_morphos_backend()
        morphos_backend.register_morphos_backend()
        assert "morphos" in torch._dynamo.list_backends()


class TestCompilation:
    def test_readme_one_liner_actually_works(self):
        """The flagship README example: torch.compile(model, backend="morphos")."""
        model = torch.nn.Linear(10, 5)
        compiled = torch.compile(model, backend="morphos")
        out = compiled(torch.randn(4, 10))
        assert out.shape == (4, 5)

    def test_compiled_output_matches_eager(self):
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4))
        x = torch.randn(3, 8)

        eager_out = model(x)
        compiled_out = torch.compile(model, backend="morphos")(x)

        assert torch.allclose(eager_out, compiled_out, atol=1e-5)


class TestAudit:
    def test_ordinary_linear_is_not_flagged(self):
        """Regression test: quantized::linear must not collide with nn.Linear."""
        model = torch.nn.Linear(10, 5)
        torch.compile(model, backend="morphos")(torch.randn(2, 10))
        assert morphos_backend.LAST_AUDIT == {}

    def test_known_slow_op_is_flagged(self):
        def f(x):
            return torch.where(x > 0, x, -x).sum()

        torch.compile(f, backend="morphos")(torch.tensor([1.0, -2.0, 3.0]))

        assert "where" in morphos_backend.LAST_AUDIT
        assert morphos_backend.LAST_AUDIT["where"]["kind"] == "slow"

    def test_audit_graph_classifies_by_namespace(self):
        # A synthetic FX-like node whose bare name collides across
        # namespaces must only match its own namespace.
        class FakeNode:
            def __init__(self, op, target):
                self.op = op
                self.target = target

        class FakeOverloadPacket:
            def __init__(self, s):
                self._s = s

            def __str__(self):
                return self._s

        linear_node = FakeNode("call_function", type("T", (), {"__name__": "linear"})())
        namespace, basename = morphos_backend._op_qualname(linear_node)
        assert (namespace, basename) == ("aten", "linear")
        # "aten::linear" is not in the unsupported table, so this must not
        # be classified as a fallback even though "quantized::linear" is.
        findings = morphos_backend._audit_graph(
            type("GM", (), {"graph": type("G", (), {"nodes": [linear_node]})()})()
        )
        assert findings == {}


class TestDelegation:
    def test_never_delegates_to_itself(self, monkeypatch):
        monkeypatch.setattr(
            "ascend_compat.cuda_shim.compile_helpers.get_compile_backend",
            lambda: "morphos",
        )

        model = torch.nn.Linear(4, 2)
        # Would hang/recurse if the self-delegation guard were broken.
        compiled = torch.compile(model, backend="morphos")
        out = compiled(torch.randn(1, 4))
        assert out.shape == (1, 2)
