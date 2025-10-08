# 损失权重调度器使用指南

## 概述

损失权重调度器是一个强大的工具，可以在训练过程中动态调整各个损失分量的权重。这对于阴影去除任务特别有用，因为不同的训练阶段可能需要关注不同的损失分量。

## 主要特性

- 🎯 **多种调度策略**: 支持线性、余弦、指数、阶梯式等调度方式
- 🔄 **自适应调整**: 根据验证指标自动调整权重
- 📊 **预设配置**: 提供针对阴影去除任务优化的预设配置
- 🛠️ **高度可定制**: 支持完全自定义的调度策略
- 📈 **可视化支持**: 可以绘制权重变化曲线

## 快速开始

### 1. 在配置文件中启用

在 `config.yml` 中添加以下配置：

```yaml
LOSS_SCHEDULER:
  ENABLE: true                    # 启用损失调度器
  PRESET: 'shadow_removal'        # 使用阴影去除预设
  ADAPTIVE_PATIENCE: 5            # 自适应调整耐心值
  ADAPTIVE_FACTOR: 0.8            # 自适应调整因子
  VERBOSE: true                   # 打印调整信息
```

### 2. 在训练脚本中集成

```python
from utils.loss_scheduler import create_loss_scheduler_from_config

# 创建损失调度器
loss_scheduler = None
if getattr(opt.LOSS_SCHEDULER, 'ENABLE', False):
    loss_scheduler = create_loss_scheduler_from_config(opt, opt.OPTIM.NUM_EPOCHS)

# 在训练循环中使用
for epoch in range(start_epoch, total_epochs + 1):
    # 更新损失权重
    if loss_scheduler is not None:
        updated_weights = loss_scheduler.step(epoch, validation_rmse)
        criterion.update_weights(updated_weights)
    
    # 正常的训练代码...
```

## 预设配置

### 1. Shadow Removal (推荐)
专为阴影去除任务优化的配置：
- MSE保持恒定作为基础损失
- SSIM逐渐增强，保持结构完整性
- Edge权重采用余弦调度，重点处理边缘问题
- Gradient线性增长，确保平滑过渡
- Boundary在中期达到峰值

```yaml
LOSS_SCHEDULER:
  PRESET: 'shadow_removal'
```

### 2. Fine Tuning
适用于微调阶段，更注重细节优化：
- 更激进的边缘和梯度权重增长
- 感知损失逐渐衰减，避免过拟合

```yaml
LOSS_SCHEDULER:
  PRESET: 'fine_tuning'
```

### 3. Aggressive
快速收敛配置，适合数据充足的情况：
- 所有权重都较高
- 快速达到最终权重值

```yaml
LOSS_SCHEDULER:
  PRESET: 'aggressive'
```

## 自定义配置

### 基本自定义

可以覆盖预设配置中的特定损失权重：

```yaml
LOSS_SCHEDULER:
  PRESET: 'shadow_removal'
  OVERRIDES:
    edge:                         # 自定义边缘损失调度
      schedule_type: "linear"
      initial_weight: 0.3
      final_weight: 1.0
    ssim:                         # 自定义SSIM损失调度
      schedule_type: "cosine"
      initial_weight: 0.1
      final_weight: 0.6
```

### 完全自定义

```python
custom_configs = {
    "mse": {
        "schedule_type": "constant",
        "initial_weight": 1.0,
        "final_weight": 1.0
    },
    "edge": {
        "schedule_type": "warmup_cosine",
        "initial_weight": 0.2,
        "final_weight": 0.8,
        "warmup_epochs": 15
    }
}

scheduler = LossWeightScheduler(
    total_epochs=100,
    loss_configs=custom_configs
)
```

## 调度策略详解

### 1. Constant (恒定)
权重保持不变
```python
{
    "schedule_type": "constant",
    "initial_weight": 0.5
}
```

### 2. Linear (线性)
权重线性变化
```python
{
    "schedule_type": "linear",
    "initial_weight": 0.2,
    "final_weight": 0.8
}
```

### 3. Cosine (余弦)
权重按余弦函数变化，开始快速变化，后期趋于稳定
```python
{
    "schedule_type": "cosine",
    "initial_weight": 0.5,
    "final_weight": 1.0
}
```

### 4. Exponential (指数)
权重指数衰减
```python
{
    "schedule_type": "exponential",
    "initial_weight": 1.0,
    "decay_rate": 0.95
}
```

### 5. Step (阶梯)
权重阶梯式下降
```python
{
    "schedule_type": "step",
    "initial_weight": 1.0,
    "step_size": 30,
    "step_gamma": 0.5
}
```

### 6. Warmup Cosine (预热余弦)
先线性预热，再余弦调度
```python
{
    "schedule_type": "warmup_cosine",
    "initial_weight": 0.1,
    "final_weight": 0.8,
    "warmup_epochs": 20
}
```

## 自适应调整

当验证指标连续多个epoch没有改善时，调度器会自动调整权重：

- **增强边缘损失**: 提高边缘处理能力
- **增强梯度损失**: 改善平滑过渡
- **降低感知损失**: 避免过拟合

```yaml
LOSS_SCHEDULER:
  ADAPTIVE_PATIENCE: 5      # 连续5个epoch无改善就调整
  ADAPTIVE_FACTOR: 0.8      # 调整因子
```

## 最佳实践

### 1. 阴影去除任务推荐配置

```yaml
LOSS_SCHEDULER:
  ENABLE: true
  PRESET: 'shadow_removal'
  ADAPTIVE_PATIENCE: 5
  ADAPTIVE_FACTOR: 0.8
  VERBOSE: true
  
  OVERRIDES:
    edge:
      schedule_type: "cosine"
      initial_weight: 0.4
      final_weight: 1.0
```

### 2. 训练阶段建议

- **早期 (1-20% epochs)**: 重点MSE和SSIM，建立基础
- **中期 (20-70% epochs)**: 增强边缘和梯度损失，处理细节
- **后期 (70-100% epochs)**: 平衡所有损失，精细调优

### 3. 监控指标

- 观察RMSE、PSNR、SSIM的变化趋势
- 注意权重调整对训练稳定性的影响
- 如果训练不稳定，降低权重变化幅度

## 故障排除

### 1. 训练不稳定
- 降低权重变化速度（使用更温和的调度策略）
- 增加warmup epochs
- 降低最终权重值

### 2. 收敛缓慢
- 使用更激进的预设配置
- 提前开始边缘损失的增长
- 降低自适应调整的patience

### 3. 过拟合
- 启用感知损失的衰减
- 降低边缘损失的最终权重
- 增加正则化损失的权重

## 示例代码

完整的使用示例请参考：
- `examples/loss_scheduler_example.py` - 基础使用示例
- `test_loss_scheduler.py` - 功能测试脚本
- `train.py` - 实际训练集成示例

## 可视化

调度器支持权重变化的可视化：

```python
# 在训练结束后绘制权重变化曲线
scheduler.plot_weight_evolution(save_path="weight_evolution.png")
```

这将生成一个显示所有损失权重随epoch变化的图表，帮助分析调度策略的效果。