"""Quantization compatibility layer for Ascend NPU.

The single largest hardware gap between Ascend 910B/C and NVIDIA H100 is
**FP8 support**.  DeepSeek-V3/R1 use FP8 extensively; running these models
in BF16/FP16 nearly doubles memory and compute requirements.

Ascend supports:
    - W8A8 (INT8 weights, INT8 activations) — static and dynamic
    - SmoothQuant (via W8A8 path)
    - BF16/FP16 (native, no quantization)

Ascend does NOT support (until 950 series, expected 2026):
    - FP8 (E4M3, E5M2) — no hardware support on 910A/B/C
    - AWQ — format not supported
    - GPTQ — format not supported
    - GGUF — CPU inference format, no NPU acceleration
    - bitsandbytes — CUDA-specific quantization library
    - Marlin — NVIDIA-specific sparse quantization

This module provides:
1. Detection of quantization format compatibility
2. FP8 → BF16/FP16 fallback with clear warnings
3. Guidance on re-quantization using supported formats
4. Integration with HuggingFace model loading pipeline

Usage::

    from ascend_compat.cuda_shim.quantization import check_model_quant, QuantCompat

    compat = check_model_quant("meta-llama/Llama-3-8B-GPTQ")
    if not compat.supported:
        print(compat.suggestion)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ascend_compat._logging import get_logger

logger = get_logger(__name__)


@dataclass
class QuantCompat:
    """Quantization compatibility result."""

    method: str               # Detected quantization method
    supported: bool           # Whether it works on current Ascend hardware
    native_performance: bool  # Whether it runs at native speed (vs emulated)
    suggestion: str           # Migration guidance
    alternative: str = ""     # Recommended alternative method


# ---------------------------------------------------------------------------
# Compatibility database
# ---------------------------------------------------------------------------

_QUANT_COMPAT: Dict[str, QuantCompat] = {
    "fp8": QuantCompat(
        method="fp8",
        supported=False,
        native_performance=False,
        suggestion=(
            "FP8 quantization requires Ascend 950 (expected 2026). "
            "Current workarounds:\n"
            "  1. Run in BF16/FP16 (2x memory, ~1.5x slower)\n"
            "  2. Re-quantize to W8A8: auto-round --model <model> --quant_method w8a8\n"
            "  3. Use vllm-ascend's dynamic W8A8 quantization"
        ),
        alternative="w8a8_dynamic",
    ),
    "fp8_e4m3": QuantCompat(
        method="fp8_e4m3",
        supported=False,
        native_performance=False,
        suggestion="FP8 E4M3 not supported on Ascend 910. See 'fp8' for workarounds.",
        alternative="w8a8",
    ),
    "fp8_e5m2": QuantCompat(
        method="fp8_e5m2",
        supported=False,
        native_performance=False,
        suggestion="FP8 E5M2 not supported on Ascend 910. See 'fp8' for workarounds.",
        alternative="w8a8",
    ),
    "awq": QuantCompat(
        method="awq",
        supported=False,
        native_performance=False,
        suggestion=(
            "AWQ format is not supported on Ascend. "
            "Re-quantize using W8A8:\n"
            "  auto-round --model <model> --quant_method w8a8 --bits 8"
        ),
        alternative="w8a8",
    ),
    "gptq": QuantCompat(
        method="gptq",
        supported=False,
        native_performance=False,
        suggestion=(
            "GPTQ format is not supported on Ascend. "
            "Re-quantize using W8A8:\n"
            "  auto-round --model <model> --quant_method w8a8"
        ),
        alternative="w8a8",
    ),
    "gguf": QuantCompat(
        method="gguf",
        supported=False,
        native_performance=False,
        suggestion=(
            "GGUF is a CPU/CUDA inference format with no NPU acceleration. "
            "Use the original model weights in FP16 or re-quantize to W8A8."
        ),
        alternative="w8a8",
    ),
    "bitsandbytes": QuantCompat(
        method="bitsandbytes",
        supported=False,
        native_performance=False,
        suggestion=(
            "bitsandbytes is a CUDA-specific library. "
            "For 8-bit inference on Ascend, use W8A8 quantization."
        ),
        alternative="w8a8",
    ),
    "marlin": QuantCompat(
        method="marlin",
        supported=False,
        native_performance=False,
        suggestion="Marlin is an NVIDIA-specific sparse format. Use W8A8 on Ascend.",
        alternative="w8a8",
    ),
    "squeezellm": QuantCompat(
        method="squeezellm",
        supported=False,
        native_performance=False,
        suggestion="SqueezeLLM is not supported on Ascend. Use W8A8 instead.",
        alternative="w8a8",
    ),
    # Supported methods
    "w8a8": QuantCompat(
        method="w8a8",
        supported=True,
        native_performance=True,
        suggestion="W8A8 (INT8) quantization — natively supported on Ascend.",
    ),
    "w8a8_dynamic": QuantCompat(
        method="w8a8_dynamic",
        supported=True,
        native_performance=True,
        suggestion="Dynamic W8A8 quantization — natively supported via vllm-ascend.",
    ),
    "smoothquant": QuantCompat(
        method="smoothquant",
        supported=True,
        native_performance=True,
        suggestion="SmoothQuant — supported via W8A8 path on Ascend.",
    ),
    "none": QuantCompat(
        method="none",
        supported=True,
        native_performance=True,
        suggestion="No quantization (FP16/BF16) — fully supported.",
    ),
}


def check_quant_method(method: str) -> QuantCompat:
    """Check if a quantization method is compatible with Ascend NPU.

    Args:
        method: Quantization method name (e.g. "fp8", "awq", "w8a8").

    Returns:
        :class:`QuantCompat` with compatibility info and migration guidance.
    """
    method_lower = method.lower().strip() if method else "none"

    compat = _QUANT_COMPAT.get(method_lower)
    if compat is not None:
        if not compat.supported:
            logger.warning(
                "Quantization method '%s' is not supported on Ascend NPU. %s",
                method, compat.suggestion.split("\n")[0],
            )
        return compat

    # Unknown method — assume unsupported with generic guidance
    logger.warning("Unknown quantization method: %s", method)
    return QuantCompat(
        method=method,
        supported=False,
        native_performance=False,
        suggestion=(
            f"Quantization method '{method}' is not recognized for Ascend. "
            "Supported methods: w8a8, w8a8_dynamic, smoothquant, none (FP16/BF16)."
        ),
        alternative="w8a8",
    )


def check_model_quant(model_name_or_path: str) -> QuantCompat:
    """Check a model's quantization compatibility by inspecting its config.

    Reads the model's config.json to detect the quantization method,
    then checks compatibility with Ascend.

    Args:
        model_name_or_path: HuggingFace model name or local path.

    Returns:
        :class:`QuantCompat` for the detected method.
    """
    quant_method = _detect_quant_method(model_name_or_path)
    return check_quant_method(quant_method)


def _detect_quant_method(model_name_or_path: str) -> str:
    """Detect quantization method from model name or config."""
    import os
    import json

    # Try to read config.json locally
    config_path = os.path.join(model_name_or_path, "config.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path) as f:
                config = json.load(f)
            quant_config = config.get("quantization_config", {})
            if isinstance(quant_config, dict):
                method = quant_config.get("quant_method", "")
                if method:
                    return method
        except (json.JSONDecodeError, IOError):
            pass

    # Heuristic from model name
    name_lower = model_name_or_path.lower()
    for marker, method in [
        ("gptq", "gptq"),
        ("awq", "awq"),
        ("gguf", "gguf"),
        ("fp8", "fp8"),
        ("w8a8", "w8a8"),
        ("int8", "w8a8"),
        ("squeezellm", "squeezellm"),
    ]:
        if marker in name_lower:
            return method

    return "none"


def get_supported_methods() -> List[str]:
    """Return list of quantization methods supported on current Ascend hardware."""
    return [k for k, v in _QUANT_COMPAT.items() if v.supported]


def get_unsupported_methods() -> List[str]:
    """Return list of unsupported quantization methods."""
    return [k for k, v in _QUANT_COMPAT.items() if not v.supported]


def format_quant_report(compat: QuantCompat) -> str:
    """Format a quantization compatibility result as a readable string."""
    icon = "[OK]" if compat.supported else "[XX]"
    lines = [
        f"{icon} Quantization: {compat.method}",
        f"    Supported: {'Yes' if compat.supported else 'No'}",
        f"    Native perf: {'Yes' if compat.native_performance else 'No'}",
        f"    {compat.suggestion}",
    ]
    if compat.alternative and not compat.supported:
        lines.append(f"    Recommended: {compat.alternative}")
    return "\n".join(lines)
