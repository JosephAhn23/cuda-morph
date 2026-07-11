"""System info and shim status. Used by cuda-morph info."""

from __future__ import annotations


def show_info() -> str:
    """Show detected hardware and shim status."""
    lines = ["cuda-morph system info", "=" * 50]

    try:
        import torch
        lines.append(f"PyTorch version:     {torch.__version__}")
    except ImportError:
        lines.append("PyTorch:             NOT INSTALLED")
        return "\n".join(lines)

    try:
        import ascend_compat
        lines.append(f"cuda-morph version:  {ascend_compat.__version__}")
        lines.append(f"Shim activated:      {ascend_compat.is_activated()}")
        lines.append(f"Preferred backend:   {ascend_compat.preferred_backend().value}")
        lines.append(f"Has Ascend NPU:      {ascend_compat.has_npu()}")
        lines.append(f"Has Cambricon MLU:   {ascend_compat.has_mlu()}")
        lines.append(f"Has AMD ROCm:        {ascend_compat.has_rocm()}")
        lines.append(f"Has Intel XPU:       {ascend_compat.has_xpu()}")
        lines.append(f"Has NVIDIA CUDA:     {ascend_compat.has_cuda()}")
        backends = ascend_compat.detect_backends()
        lines.append(f"All backends:        {[b.value for b in backends]}")
    except Exception as e:
        lines.append(f"Error loading cuda_morph: {e}")

    # Show registered backend details
    lines.append("")
    lines.append("Registered backends:")
    try:
        from ascend_compat.backends import BACKEND_REGISTRY
        for name, cls in BACKEND_REGISTRY.items():
            version = cls.get_adapter_version()
            available = cls.is_available()
            icon = "[OK]" if available else "[--]"
            ver_str = version if version else "not installed"
            lines.append(
                f"  {icon} {cls.display_name:<25} "
                f"adapter={cls.adapter_module:<12} ({ver_str})"
            )
    except Exception:
        pass

    return "\n".join(lines)
