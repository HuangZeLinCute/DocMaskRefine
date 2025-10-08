#!/usr/bin/env python3
"""
损失权重调度器使用示例
演示如何在训练循环中集成动态损失权重调整
"""

import torch
import torch.nn as nn
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.loss_scheduler import LossWeightScheduler, PresetLossSchedulers
from utils.losses import ReconstructionLoss


def example_training_with_scheduler():
    """演示如何在训练中使用损失调度器"""
    
    print("📚 损失权重调度器使用示例")
    print("=" * 50)
    
    # 1. 设置训练参数
    total_epochs = 50
    batch_size = 2
    image_size = 256
    
    # 2. 创建损失调度器配置
    loss_configs = PresetLossSchedulers.get_shadow_removal_config(total_epochs)
    
    # 3. 初始化损失调度器
    scheduler = LossWeightScheduler(
        total_epochs=total_epochs,
        loss_configs=loss_configs,
        adaptive_patience=3,
        adaptive_factor=0.9,
        verbose=True
    )
    
    # 4. 初始化损失函数（使用初始权重）
    initial_weights = scheduler.get_current_weights()
    criterion = ReconstructionLoss(
        mse_weight=initial_weights.get('mse', 1.0),
        ssim_weight=initial_weights.get('ssim', 0.2),
        edge_weight=initial_weights.get('edge', 0.3),
        gradient_weight=initial_weights.get('gradient', 0.1),
        boundary_weight=initial_weights.get('boundary', 0.5),
        transparency_weight=initial_weights.get('transparency', 0.1),
        perceptual_weight=initial_weights.get('perceptual', 0.1),
        histogram_weight=initial_weights.get('histogram', 0.05)
    )
    
    print(f"\n🎯 初始损失权重:")
    for name, weight in initial_weights.items():
        print(f"   {name:12s}: {weight:.4f}")
    
    # 5. 模拟训练循环
    print(f"\n🚀 开始模拟训练 ({total_epochs} epochs)")
    print("-" * 50)
    
    best_rmse = float('inf')
    
    for epoch in range(1, total_epochs + 1):
        # 模拟验证RMSE（逐渐改善，偶有波动）
        base_rmse = 0.4 - (epoch / total_epochs) * 0.25
        noise = 0.02 * torch.randn(1).item()
        current_rmse = max(0.1, base_rmse + noise)
        
        # 更新最佳RMSE
        if current_rmse < best_rmse:
            best_rmse = current_rmse
        
        # 更新损失权重
        updated_weights = scheduler.step(epoch, current_rmse)
        criterion.update_weights(updated_weights)
        
        # 模拟一个训练batch
        pred = torch.randn(batch_size, 3, image_size, image_size)
        target = torch.randn(batch_size, 3, image_size, image_size)
        
        # 计算损失
        total_loss, loss_mse, loss_ssim, loss_edge, loss_gradient, \
        loss_boundary, loss_transparency, loss_perceptual, loss_histogram = criterion(pred, target)
        
        # 打印训练信息
        if epoch % 10 == 0 or epoch <= 5:
            print(f"\nEpoch {epoch:2d}:")
            print(f"   RMSE: {current_rmse:.4f} (best: {best_rmse:.4f})")
            print(f"   Total Loss: {total_loss.item():.4f}")
            print(f"   Components: MSE={loss_mse.item():.3f}, SSIM={loss_ssim.item():.3f}, "
                  f"Edge={loss_edge.item():.3f}, Grad={loss_gradient.item():.3f}")
            
            # 显示当前权重
            current_weights = criterion.get_weights()
            weights_str = ", ".join([f"{k}:{v:.3f}" for k, v in current_weights.items()])
            print(f"   Weights: {weights_str}")
    
    print(f"\n✅ 训练完成!")
    print(f"📊 最终权重:")
    final_weights = criterion.get_weights()
    for name, weight in final_weights.items():
        initial_weight = initial_weights.get(name, 0.0)
        change = weight - initial_weight
        change_str = f"({change:+.3f})" if change != 0 else ""
        print(f"   {name:12s}: {weight:.4f} {change_str}")


def example_custom_scheduler():
    """演示如何创建自定义调度器"""
    
    print("\n" + "=" * 50)
    print("🛠️  自定义损失调度器示例")
    print("=" * 50)
    
    # 自定义损失配置
    custom_loss_configs = {
        "mse": {
            "schedule_type": "constant",
            "initial_weight": 1.0,
            "final_weight": 1.0
        },
        "ssim": {
            "schedule_type": "linear",
            "initial_weight": 0.1,
            "final_weight": 0.5
        },
        "edge": {
            "schedule_type": "cosine",
            "initial_weight": 0.2,
            "final_weight": 0.8
        },
        "gradient": {
            "schedule_type": "warmup_cosine",
            "initial_weight": 0.05,
            "final_weight": 0.4,
            "warmup_epochs": 10
        }
    }
    
    # 创建自定义调度器
    scheduler = LossWeightScheduler(
        total_epochs=30,
        loss_configs=custom_loss_configs,
        verbose=True
    )
    
    print("\n📈 权重变化演示:")
    test_epochs = [1, 5, 10, 15, 20, 25, 30]
    
    for epoch in test_epochs:
        weights = scheduler.step(epoch)
        print(f"\nEpoch {epoch:2d}:")
        for loss_name, weight in weights.items():
            print(f"   {loss_name:8s}: {weight:.4f}")


def example_adaptive_adjustment():
    """演示自适应权重调整"""
    
    print("\n" + "=" * 50)
    print("🔄 自适应权重调整示例")
    print("=" * 50)
    
    # 创建调度器
    loss_configs = {
        "mse": {"schedule_type": "constant", "initial_weight": 1.0},
        "edge": {"schedule_type": "constant", "initial_weight": 0.5},
        "perceptual": {"schedule_type": "constant", "initial_weight": 0.2}
    }
    
    scheduler = LossWeightScheduler(
        total_epochs=20,
        loss_configs=loss_configs,
        adaptive_patience=3,  # 3个epoch无改善就调整
        adaptive_factor=0.8,
        verbose=True
    )
    
    print("\n🎭 模拟验证指标停滞情况:")
    
    # 模拟RMSE停滞不前的情况
    rmse_values = [0.3, 0.28, 0.26, 0.25, 0.25, 0.26, 0.25, 0.24, 0.24, 0.24, 0.23]
    
    for epoch, rmse in enumerate(rmse_values, 1):
        weights = scheduler.step(epoch, rmse)
        print(f"\nEpoch {epoch:2d} - RMSE: {rmse:.3f}")
        
        if epoch >= 4:  # 从第4个epoch开始可能触发自适应调整
            edge_weight = weights.get('edge', 0.5)
            perceptual_weight = weights.get('perceptual', 0.2)
            print(f"   Edge权重: {edge_weight:.4f}, Perceptual权重: {perceptual_weight:.4f}")


if __name__ == '__main__':
    # 运行所有示例
    example_training_with_scheduler()
    example_custom_scheduler()
    example_adaptive_adjustment()
    
    print("\n" + "=" * 50)
    print("🎉 所有示例运行完成!")
    print("💡 提示: 在实际训练中，将这些代码集成到你的训练循环中即可")
    print("=" * 50)