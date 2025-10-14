"""
简化的损失函数调度器 - 专用于微调阶段的MSE+SSIM权重调整
基于RMSE值动态调整权重比例，优化模型性能
"""

import torch
import math
from typing import Dict, Optional
from enum import Enum


class ScheduleType(Enum):
    """调度器类型枚举"""
    LINEAR = "linear"
    CONSTANT = "constant"
    RMSE_ADAPTIVE = "rmse_adaptive"


class LossWeightScheduler:
    """
    简化的损失权重调度器 - 专用于微调阶段
    
    只支持MSE和SSIM两个损失的权重调度，
    基于RMSE值动态调整权重比例
    """
    
    def __init__(self, 
                 total_epochs: int,
                 mse_weight: float = 1.0,
                 ssim_weight: float = 0.3,
                 adaptive_patience: int = 3,
                 adaptive_factor: float = 0.8,
                 verbose: bool = True):
        """
        Args:
            total_epochs: 总训练轮数
            mse_weight: MSE损失初始权重
            ssim_weight: SSIM损失初始权重
            adaptive_patience: 自适应调整的耐心值
            adaptive_factor: 自适应调整因子
            verbose: 是否打印调整信息
        """
        self.total_epochs = total_epochs
        self.adaptive_patience = adaptive_patience
        self.adaptive_factor = adaptive_factor
        self.verbose = verbose
        
        # 当前权重 - 只支持MSE和SSIM
        self.current_weights = {
            'mse': mse_weight,
            'ssim': ssim_weight
        }
        
        # 自适应调整相关
        self.best_metric = float('inf')
        self.patience_counter = 0
        self.metric_history = []
        
        if self.verbose:
            print("🎯 微调专用损失权重调度器初始化完成")
            print(f"   初始权重: MSE={mse_weight:.3f}, SSIM={ssim_weight:.3f}")
    

    
    def get_current_weights(self) -> Dict[str, float]:
        """获取当前权重"""
        return self.current_weights.copy()
    
    def step(self, epoch: int, validation_metric: Optional[float] = None) -> Dict[str, float]:
        """
        更新权重 - 基于RMSE动态调整MSE和SSIM权重
        
        Args:
            epoch: 当前epoch (1-based)
            validation_metric: 验证指标 (RMSE，越小越好)
            
        Returns:
            updated_weights: 更新后的权重字典
        """
        # 基于RMSE调整权重
        if validation_metric is not None:
            self._rmse_based_adjustment(validation_metric, epoch)
            self._adaptive_adjustment(validation_metric, epoch)
        
        # 打印权重变化
        if self.verbose and epoch % 10 == 0:
            self._print_current_weights(epoch, validation_metric)
        
        return self.get_current_weights()
    
    def _rmse_based_adjustment(self, current_rmse: float, epoch: int):
        """基于RMSE值调整MSE和SSIM权重"""
        if current_rmse > 0.15:
            # RMSE较高，优先MSE快速收敛
            self.current_weights['mse'] = 1.2
            self.current_weights['ssim'] = 0.2
        elif current_rmse > 0.12:
            # RMSE中等，平衡优化
            self.current_weights['mse'] = 1.0
            self.current_weights['ssim'] = 0.4
        elif current_rmse > 0.10:
            # RMSE较好，增强结构保持
            self.current_weights['mse'] = 0.9
            self.current_weights['ssim'] = 0.5
        else:
            # RMSE很好，重点保持质量
            self.current_weights['mse'] = 0.8
            self.current_weights['ssim'] = 0.6
    
    def _adaptive_adjustment(self, validation_metric: float, epoch: int):
        """基于验证指标的自适应调整"""
        self.metric_history.append(validation_metric)
        
        # 检查是否有改善
        if validation_metric < self.best_metric:
            self.best_metric = validation_metric
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # 如果连续多个epoch没有改善，微调权重
        if self.patience_counter >= self.adaptive_patience:
            self._perform_adaptive_adjustment(epoch)
            self.patience_counter = 0
    
    def _perform_adaptive_adjustment(self, epoch: int):
        """执行自适应权重调整 - 只针对MSE和SSIM"""
        if self.verbose:
            print(f"\n🔄 Epoch {epoch}: 执行自适应权重调整")
        
        # 策略：当性能停滞时，适度增强SSIM权重，降低MSE权重
        old_mse = self.current_weights["mse"]
        old_ssim = self.current_weights["ssim"]
        
        # 降低MSE权重，避免过拟合
        self.current_weights["mse"] = max(old_mse * self.adaptive_factor, 0.5)
        
        # 增强SSIM权重，改善结构相似性
        self.current_weights["ssim"] = min(old_ssim * 1.2, 0.8)
        
        if self.verbose:
            print(f"   MSE权重: {old_mse:.3f} → {self.current_weights['mse']:.3f}")
            print(f"   SSIM权重: {old_ssim:.3f} → {self.current_weights['ssim']:.3f}")
    
    def _print_current_weights(self, epoch: int, rmse: Optional[float] = None):
        """打印当前权重"""
        rmse_str = f" (RMSE: {rmse:.4f})" if rmse is not None else ""
        print(f"\n📊 Epoch {epoch}{rmse_str} 当前损失权重:")
        print(f"   MSE: {self.current_weights['mse']:.4f}")
        print(f"   SSIM: {self.current_weights['ssim']:.4f}")
    
    def get_schedule_summary(self) -> str:
        """获取调度器摘要信息"""
        summary = "Fine-Tuning Loss Weight Scheduler Summary:\n"
        summary += f"Total Epochs: {self.total_epochs}\n"
        summary += f"Adaptive Patience: {self.adaptive_patience}\n"
        summary += f"Current Weights: MSE={self.current_weights['mse']:.3f}, SSIM={self.current_weights['ssim']:.3f}\n"
        summary += f"Best RMSE: {self.best_metric:.4f}\n"
        
        return summary
    
    def save_state(self, filepath: str):
        """保存调度器状态"""
        state = {
            'current_weights': self.current_weights,
            'best_metric': self.best_metric,
            'patience_counter': self.patience_counter,
            'metric_history': self.metric_history
        }
        torch.save(state, filepath)
    
    def load_state(self, filepath: str):
        """加载调度器状态"""
        state = torch.load(filepath, map_location='cpu')
        self.current_weights = state.get('current_weights', {})
        self.best_metric = state.get('best_metric', float('inf'))
        self.patience_counter = state.get('patience_counter', 0)
        self.metric_history = state.get('metric_history', [])


class FineTuningPresets:
    """微调阶段的预设配置"""
    
    @staticmethod
    def get_balanced_config() -> Dict[str, float]:
        """
        平衡微调配置 - MSE和SSIM权重平衡
        适合大多数微调场景
        """
        return {
            "mse_weight": 1.0,
            "ssim_weight": 0.3
        }
    
    @staticmethod
    def get_quality_focused_config() -> Dict[str, float]:
        """
        质量优先配置 - 更注重SSIM结构相似性
        适合RMSE已经较低，需要提升视觉质量的场景
        """
        return {
            "mse_weight": 0.8,
            "ssim_weight": 0.5
        }
    
    @staticmethod
    def get_rmse_focused_config() -> Dict[str, float]:
        """
        RMSE优先配置 - 更注重MSE像素级匹配
        适合RMSE还需要进一步降低的场景
        """
        return {
            "mse_weight": 1.2,
            "ssim_weight": 0.2
        }


def create_loss_scheduler_from_config(config, total_epochs: int) -> LossWeightScheduler:
    """
    从配置文件创建微调专用损失调度器
    
    Args:
        config: 配置对象，包含LOSS_SCHEDULER部分
        total_epochs: 总训练轮数
        
    Returns:
        LossWeightScheduler实例
    """
    try:
        # 使用默认的平衡微调配置
        preset_config = FineTuningPresets.get_balanced_config()
        
        # 创建调度器 - 使用简化的默认配置
        scheduler = LossWeightScheduler(
            total_epochs=total_epochs,
            mse_weight=preset_config['mse_weight'],
            ssim_weight=preset_config['ssim_weight'],
            adaptive_patience=3,        # 固定为3
            adaptive_factor=0.8,       # 固定为0.8
            verbose=True               # 固定为True
        )
        
        return scheduler
        
    except Exception as e:
        print(f"❌ 创建损失调度器失败: {e}")
        # 返回一个简单的默认调度器
        return LossWeightScheduler(
            total_epochs=total_epochs,
            mse_weight=1.0,
            ssim_weight=0.3,
            adaptive_patience=3,
            adaptive_factor=0.8,
            verbose=True
        )


if __name__ == '__main__':
    # 测试微调专用损失调度器
    print("🧪 测试微调专用损失权重调度器")
    
    # 创建调度器
    scheduler = LossWeightScheduler(
        total_epochs=50,
        mse_weight=1.0,
        ssim_weight=0.3,
        verbose=True
    )
    
    # 模拟微调过程
    print("\n🚀 模拟微调过程:")
    test_epochs = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    test_rmse = [0.18, 0.16, 0.14, 0.13, 0.12, 0.11, 0.105, 0.102, 0.100, 0.098, 0.097]
    
    for epoch, rmse in zip(test_epochs, test_rmse):
        weights = scheduler.step(epoch, rmse)
        print(f"\nEpoch {epoch:3d} (RMSE: {rmse:.3f}):")
        print(f"  MSE:  {weights['mse']:.4f}")
        print(f"  SSIM: {weights['ssim']:.4f}")
    
    print(f"\n📊 调度器摘要:")
    print(scheduler.get_schedule_summary())
