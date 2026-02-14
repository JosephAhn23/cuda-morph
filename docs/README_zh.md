# cuda-morph (中文)

**两行代码，零修改，在非NVIDIA加速器上运行PyTorch CUDA代码。**

```python
import ascend_compat
ascend_compat.activate()

# 以下代码无需修改，自动路由到检测到的后端
model = AutoModelForCausalLM.from_pretrained("gpt2").cuda()
output = model.generate(**inputs)
```

## 问题

全球 AI 软件生态被锁定在 NVIDIA CUDA 上。HuggingFace、DeepSpeed、vLLM、flash-attn 以及成千上万的训练脚本都硬编码了 `torch.cuda`。在非 NVIDIA 硬件上运行现有代码通常需要数周的手动移植。

cuda-morph 在 Python 层拦截 `torch.cuda` 调用，自动路由到实际存在的后端。

## 工作原理

- `torch.cuda.is_available()` → 在 NPU 上返回 `False`（防止 NCCL 误检测）
- `torch.device("cuda")` → 透明重映射为 `torch.device("npu")`
- `Tensor.cuda()` / `Module.cuda()` → 重定向到 `.npu()`
- `flash_attn` → 替换为 `npu_fusion_attention`
- DeepSpeed → 注册 HCCL 替代 NCCL

## 当前状态

> cuda-morph 已通过仿真验证，尚未通过硬件验证。
> 460+ 测试在 CPU 回退模式下通过。正在寻求硬件合作伙伴。

| 后端 | 硬件 | 状态 |
|------|------|------|
| **华为昇腾** | 910B, 310P | 完整适配 + 生态补丁，需硬件验证 |
| AMD ROCm | MI210, MI250X, MI300X | 检测 + 设备路由 |
| Intel XPU | Max 1550, Flex, Arc | 检测 + 设备路由 |
| 寒武纪 | MLU370, MLU590 | 检测 + 设备路由 |

## 快速开始

```bash
pip install cuda-morph
```

```python
import ascend_compat
ascend_compat.activate()  # 自动检测硬件，补丁 torch.cuda

# 以下代码无需修改
model = model.cuda()
```

## 定位

cuda-morph 不与 [FlagOS/FlagScale](https://github.com/FlagOpen/FlagScale) 竞争。FlagOS 是国家支持的生产级基础设施，专注于深度优化和算子生成。cuda-morph 更窄：仅做运行时适配，专注于零代码迁移。

详细战略请参见 [STRATEGY.md](../STRATEGY.md)。

## 许可证

Apache 2.0
