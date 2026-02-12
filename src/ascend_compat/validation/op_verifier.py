"""Empirical operator verification harness.

Compares the output of shimmed operators (running on NPU) against reference
CPU implementations.  Every check returns a :class:`VerificationResult` with
the numerical error bounds so you know *exactly* how accurate the shim is.

Design principles:
- No mocking.  This runs real compute on real hardware.
- Deterministic seeds for reproducibility.
- Reports max abs error, mean abs error, and cosine similarity.
- Fails loud: if an operator can't run, the result says so (not a silent skip).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch

from ascend_compat._logging import get_logger

logger = get_logger(__name__)


@dataclass
class VerificationResult:
    """Result of a single operator verification."""

    op_name: str
    passed: bool
    max_abs_error: float = 0.0
    mean_abs_error: float = 0.0
    cosine_similarity: float = 1.0
    rtol_used: float = 0.0
    atol_used: float = 0.0
    device: str = ""
    dtype: str = ""
    shape_info: str = ""
    elapsed_ms: float = 0.0
    error_message: str = ""
    notes: str = ""


class OperatorVerifier:
    """Verify shimmed operators against CPU reference implementations.

    Args:
        device: Device to test on (``"npu"`` for real hardware,
            ``"cpu"`` for simulation mode).
        seed: Random seed for reproducibility.
        dtype: Default dtype for test tensors.

    Example::

        verifier = OperatorVerifier(device="npu")
        results = verifier.run_all()
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            print(f"[{status}] {r.op_name}: max_err={r.max_abs_error:.6f}")
    """

    def __init__(
        self,
        device: str = "npu",
        seed: int = 42,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.device = device
        self.seed = seed
        self.dtype = dtype

    def _to_device(self, t: torch.Tensor) -> torch.Tensor:
        """Move tensor to test device."""
        if self.device == "npu":
            return t.npu()
        elif self.device == "cuda":
            return t.cuda()
        return t

    def _compare(
        self,
        actual: torch.Tensor,
        expected: torch.Tensor,
        rtol: float = 1e-2,
        atol: float = 1e-2,
    ) -> VerificationResult:
        """Compute numerical comparison metrics."""
        actual_f = actual.cpu().float()
        expected_f = expected.cpu().float()

        diff = (actual_f - expected_f).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()

        # Cosine similarity (flatten both tensors)
        a_flat = actual_f.reshape(-1)
        e_flat = expected_f.reshape(-1)
        cos_sim = torch.nn.functional.cosine_similarity(
            a_flat.unsqueeze(0), e_flat.unsqueeze(0)
        ).item()

        passed = max_abs <= atol or (
            max_abs <= rtol * expected_f.abs().max().item() + atol
        )

        return VerificationResult(
            op_name="",  # Caller fills this in
            passed=passed,
            max_abs_error=max_abs,
            mean_abs_error=mean_abs,
            cosine_similarity=cos_sim,
            rtol_used=rtol,
            atol_used=atol,
        )

    # -------------------------------------------------------------------
    # Individual operator verifications
    # -------------------------------------------------------------------

    def verify_flash_attention(self) -> VerificationResult:
        """Verify flash_attn_func → npu_fusion_attention mapping.

        Compares our shimmed flash_attn against naive scaled dot-product
        attention computed in float32 on CPU.  This is the single most
        important verification in the entire project.
        """
        from ascend_compat.ecosystem.flash_attn import flash_attn_func

        torch.manual_seed(self.seed)
        batch, seqlen, nheads, headdim = 2, 64, 4, 32

        q = torch.randn(batch, seqlen, nheads, headdim, dtype=self.dtype)
        k = torch.randn(batch, seqlen, nheads, headdim, dtype=self.dtype)
        v = torch.randn(batch, seqlen, nheads, headdim, dtype=self.dtype)

        # Reference: naive attention in float32
        scale = 1.0 / math.sqrt(headdim)
        q_ref = q.float().permute(0, 2, 1, 3)
        k_ref = k.float().permute(0, 2, 1, 3)
        v_ref = v.float().permute(0, 2, 1, 3)
        weights = torch.softmax(q_ref @ k_ref.transpose(-2, -1) * scale, dim=-1)
        ref_output = (weights @ v_ref).permute(0, 2, 1, 3)

        # Our shim on device
        start = time.perf_counter()
        try:
            npu_output = flash_attn_func(
                self._to_device(q), self._to_device(k), self._to_device(v)
            )
        except Exception as e:
            return VerificationResult(
                op_name="flash_attn_func",
                passed=False,
                device=self.device,
                dtype=str(self.dtype),
                shape_info=f"({batch},{seqlen},{nheads},{headdim})",
                error_message=str(e),
            )
        elapsed = (time.perf_counter() - start) * 1000

        result = self._compare(npu_output, ref_output, rtol=5e-2, atol=5e-2)
        result.op_name = "flash_attn_func"
        result.device = self.device
        result.dtype = str(self.dtype)
        result.shape_info = f"({batch},{seqlen},{nheads},{headdim})"
        result.elapsed_ms = elapsed
        result.notes = "Compared against naive SDPA in float32"
        return result

    def verify_flash_attention_causal(self) -> VerificationResult:
        """Verify causal masking in flash_attn → npu_fusion_attention."""
        from ascend_compat.ecosystem.flash_attn import flash_attn_func

        torch.manual_seed(self.seed)
        batch, seqlen, nheads, headdim = 1, 32, 2, 32

        q = torch.randn(batch, seqlen, nheads, headdim, dtype=self.dtype)
        k = torch.randn(batch, seqlen, nheads, headdim, dtype=self.dtype)
        v = torch.randn(batch, seqlen, nheads, headdim, dtype=self.dtype)

        # Reference: causal attention in float32
        scale = 1.0 / math.sqrt(headdim)
        q_ref = q.float().permute(0, 2, 1, 3)
        k_ref = k.float().permute(0, 2, 1, 3)
        v_ref = v.float().permute(0, 2, 1, 3)
        scores = q_ref @ k_ref.transpose(-2, -1) * scale

        # Causal mask
        mask = torch.triu(torch.ones(seqlen, seqlen), diagonal=1).bool()
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        ref_output = (weights @ v_ref).permute(0, 2, 1, 3)

        # Our shim
        start = time.perf_counter()
        try:
            npu_output = flash_attn_func(
                self._to_device(q), self._to_device(k), self._to_device(v),
                causal=True,
            )
        except Exception as e:
            return VerificationResult(
                op_name="flash_attn_func(causal=True)",
                passed=False,
                device=self.device,
                dtype=str(self.dtype),
                error_message=str(e),
            )
        elapsed = (time.perf_counter() - start) * 1000

        result = self._compare(npu_output, ref_output, rtol=5e-2, atol=5e-2)
        result.op_name = "flash_attn_func(causal=True)"
        result.device = self.device
        result.dtype = str(self.dtype)
        result.shape_info = f"({batch},{seqlen},{nheads},{headdim})"
        result.elapsed_ms = elapsed
        result.notes = "Causal mask: upper-triangular"
        return result

    def verify_matmul(self) -> VerificationResult:
        """Verify basic matmul correctness on device."""
        torch.manual_seed(self.seed)
        a = torch.randn(64, 128, dtype=self.dtype)
        b = torch.randn(128, 64, dtype=self.dtype)

        ref = (a.float() @ b.float())

        start = time.perf_counter()
        try:
            out = self._to_device(a) @ self._to_device(b)
        except Exception as e:
            return VerificationResult(
                op_name="matmul", passed=False, device=self.device,
                error_message=str(e),
            )
        elapsed = (time.perf_counter() - start) * 1000

        result = self._compare(out, ref, rtol=1e-2, atol=1e-2)
        result.op_name = "matmul"
        result.device = self.device
        result.dtype = str(self.dtype)
        result.shape_info = "(64,128) @ (128,64)"
        result.elapsed_ms = elapsed
        return result

    def verify_softmax(self) -> VerificationResult:
        """Verify softmax correctness on device."""
        torch.manual_seed(self.seed)
        x = torch.randn(16, 256, dtype=self.dtype)

        ref = torch.softmax(x.float(), dim=-1)

        start = time.perf_counter()
        try:
            out = torch.softmax(self._to_device(x), dim=-1)
        except Exception as e:
            return VerificationResult(
                op_name="softmax", passed=False, device=self.device,
                error_message=str(e),
            )
        elapsed = (time.perf_counter() - start) * 1000

        result = self._compare(out, ref, rtol=1e-3, atol=1e-3)
        result.op_name = "softmax"
        result.device = self.device
        result.dtype = str(self.dtype)
        result.shape_info = "(16,256)"
        result.elapsed_ms = elapsed
        return result

    def verify_layer_norm(self) -> VerificationResult:
        """Verify LayerNorm correctness on device."""
        torch.manual_seed(self.seed)
        x = torch.randn(4, 32, 128, dtype=self.dtype)
        ln = torch.nn.LayerNorm(128)

        ref = ln(x.float())

        start = time.perf_counter()
        try:
            ln_dev = ln.to(self.device)
            out = ln_dev(self._to_device(x))
        except Exception as e:
            return VerificationResult(
                op_name="LayerNorm", passed=False, device=self.device,
                error_message=str(e),
            )
        elapsed = (time.perf_counter() - start) * 1000

        result = self._compare(out, ref, rtol=1e-2, atol=1e-2)
        result.op_name = "LayerNorm"
        result.device = self.device
        result.dtype = str(self.dtype)
        result.shape_info = "(4,32,128)"
        result.elapsed_ms = elapsed
        return result

    # -------------------------------------------------------------------
    # Run all verifications
    # -------------------------------------------------------------------

    def run_all(self) -> List[VerificationResult]:
        """Run all operator verifications and return results."""
        checks: List[Callable[[], VerificationResult]] = [
            self.verify_matmul,
            self.verify_softmax,
            self.verify_layer_norm,
            self.verify_flash_attention,
            self.verify_flash_attention_causal,
        ]

        results = []
        for check in checks:
            logger.info("Running %s...", check.__name__)
            result = check()
            results.append(result)
            status = "PASS" if result.passed else "FAIL"
            logger.info(
                "  [%s] %s: max_err=%.6f, cos_sim=%.6f",
                status, result.op_name, result.max_abs_error, result.cosine_similarity,
            )

        return results

    @staticmethod
    def format_report(results: List[VerificationResult]) -> str:
        """Format verification results as a human-readable report."""
        lines = [
            "Operator Verification Report",
            "=" * 70,
            f"{'Operator':<35} {'Status':<8} {'Max Err':<12} {'Cos Sim':<10} {'Time':>8}",
            "-" * 70,
        ]

        for r in results:
            status = "PASS" if r.passed else "FAIL"
            if r.error_message:
                lines.append(f"{r.op_name:<35} {'ERROR':<8} {r.error_message}")
            else:
                lines.append(
                    f"{r.op_name:<35} {status:<8} {r.max_abs_error:<12.6f} "
                    f"{r.cosine_similarity:<10.6f} {r.elapsed_ms:>7.1f}ms"
                )
                if r.notes:
                    lines.append(f"  note: {r.notes}")

        lines.append("-" * 70)
        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if not r.passed)
        lines.append(f"Total: {len(results)} | Passed: {passed} | Failed: {failed}")

        if failed > 0:
            lines.append("")
            lines.append("FAILED operators:")
            for r in results:
                if not r.passed:
                    msg = r.error_message or f"max_err={r.max_abs_error:.6f} > atol={r.atol_used}"
                    lines.append(f"  - {r.op_name}: {msg}")

        lines.append("=" * 70)
        return "\n".join(lines)
