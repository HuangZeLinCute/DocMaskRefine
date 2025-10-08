# 损失调度器 - 修复版本使用指南

## 🔧 问题修复

已修复配置文件中的 YACS 配置错误，现在损失调度器可以正常工作。

## 🚀 快速使用

### 1. 确认配置文件
确保 `config.yml` 包含以下配置：

```yaml
LOSS_SCHEDULER:
  ENABLE: true                # 启用损失调度器
  PRESET: 'shadow_removal'    # 使用阴影去除预设
  ADAPTIVE_PATIENCE: 5        # 自适应调整耐心值
  ADAPTIVE_FACTOR: 0.8        # 自适应调整因子
  VERBOSE: true               # 显示调整信息
```

### 2. 运行测试
在训练前，可以运行快速测试验证功能：

```bash
python quick_test.py
```

### 3. 开始训练
如果测试通过，直接运行训练：

```bash
python train.py
```

## 📊 预期效果

### 训练过程中的权重变化
- **Epoch 1-20**: 重点MSE和SSIM，建立基础重建能力
- **Epoch 20-70**: 逐步增强边缘和梯度损失，处理阴影边界
- **Epoch 70-100**: 平衡所有损失分量，精细调优

### 控制台输出示例
```
🎯 损失权重调度器初始化完成

📋 损失权重调度配置:
   mse         : constant        1.000 → 1.000
   ssim        : warmup_cosine   0.100 → 0.400
   edge        : cosine          0.300 → 0.800
   gradient    : linear          0.200 → 0.500
   ...

✅ 损失权重调度器已启用

epoch: 10, RMSE:0.2845, PSNR: 28.45, SSIM: 0.8234, best RMSE: 0.2845, best epoch: 10
   📊 当前损失权重: mse:1.000, ssim:0.150, edge:0.420, gradient:0.250, ...
```

## 🎯 三种预设配置

### 1. Shadow Removal (推荐)
```yaml
PRESET: 'shadow_removal'
```
- 专为阴影去除优化
- 平衡的权重变化策略
- 适合大多数文档阴影去除任务

### 2. Fine Tuning
```yaml
PRESET: 'fine_tuning'
```
- 适用于微调阶段
- 更激进的边缘处理
- 适合在预训练模型基础上微调

### 3. Aggressive
```yaml
PRESET: 'aggressive'
```
- 快速收敛配置
- 较高的损失权重
- 适合数据充足、需要快速训练的场景

## 🔄 自适应调整机制

当验证RMSE连续5个epoch没有改善时，调度器会自动：
- 增强边缘损失权重 (×1.2)
- 增强梯度损失权重 (×1.1)  
- 降低感知损失权重 (×0.8)

可以通过以下参数调整：
```yaml
ADAPTIVE_PATIENCE: 5      # 耐心值，连续多少个epoch无改善才调整
ADAPTIVE_FACTOR: 0.8      # 调整因子
```

## 📈 监控训练效果

### 关键指标
- **RMSE**: 主要优化目标，应该持续下降
- **PSNR**: 图像质量指标，应该持续上升
- **SSIM**: 结构相似性，应该持续上升

### 权重变化观察
- 每10个epoch会显示当前权重
- 观察边缘和梯度损失权重的增长
- 注意自适应调整的触发时机

## 🛠️ 故障排除

### 1. 配置加载错误
```bash
# 运行测试脚本检查配置
python quick_test.py
```

### 2. 训练不稳定
- 降低 `ADAPTIVE_FACTOR` 到 0.5-0.7
- 增加 `ADAPTIVE_PATIENCE` 到 8-10
- 考虑使用 `fine_tuning` 预设

### 3. 收敛缓慢
- 使用 `aggressive` 预设
- 降低 `ADAPTIVE_PATIENCE` 到 3
- 检查学习率设置

## 📁 相关文件

- `config.yml` - 主配置文件
- `utils/loss_scheduler.py` - 调度器实现
- `utils/losses.py` - 损失函数（支持动态权重）
- `train.py` - 训练脚本（已集成调度器）
- `quick_test.py` - 快速测试脚本

## 💡 使用建议

1. **首次使用**: 使用默认的 `shadow_removal` 预设
2. **训练监控**: 观察前20个epoch的权重变化和指标趋势
3. **参数调整**: 根据具体数据集特点调整耐心值和调整因子
4. **实验对比**: 可以尝试不同预设，找到最适合的配置

## 🎉 预期改善

使用损失调度器后，你应该能看到：
- **更好的边缘处理**: 阴影边界更加自然
- **更快的收敛**: 训练效率提升
- **更稳定的训练**: 减少训练震荡
- **更好的最终效果**: RMSE、PSNR、SSIM指标改善

---

**注意**: 如果遇到任何问题，请先运行 `python quick_test.py` 进行诊断。