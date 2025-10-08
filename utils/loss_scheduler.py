"""
损失函数调整器 - 在训练过程中动态调整各个损失分量的权重
支持多种调整策略：线性、余弦、指数、阶梯式等
"""

import torch
import math
from typing import Dict, List, Union, Optional
from enum import Enum


class ScheduleType(Enum):
    """调度器类型枚举"""
    LINEAR = "linear"
    COSINE = "cosine"
    EXPONENTIAL = "exponential"
    STEP = "step"
    WARMUP_COSINE = "warmup_cosine"
    ADAPTIVE = "adaptive"
    CONSTANT = "constant"


class LossWeightScheduler:
    """
    损失权重调度器
    
    支持为不同的损失分量设置不同的调度策略，
    可以根据训练进度、验证指标等动态调整权重
    """
    
    def __init__(self, 
                 total_epochs: int,
                 loss_configs: Dict[str, Dict],
                 adaptive_patience: int = 5,
                 adaptive_factor: float = 0.8,
                 verbose: bool = True):
        """
        Args:
            total_epochs: 总训练轮数
            loss_configs: 损失配置字典，格式如下：
                {
                    "mse": {
                        "schedule_type": "constant",
                        "initial_weight": 1.0,
                        "final_weight": 1.0
                    },
                    "ssim": {
                        "schedule_type": "warmup_cosine", 
                        "initial_weight": 0.1,
                        "final_weight": 0.5,
                        "warmup_epochs": 10
                    },
                    "edge": {
                        "schedule_type": "linear",
                        "initial_weight": 0.3,
                        "final_weight": 0.8,
                        "start_epoch": 5
                    }
                }
            adaptive_patience: 自适应调整的耐心值
            adaptive_factor: 自适应调整因子
            verbose: 是否打印调整信息
        """
        self.total_epochs = total_epochs
        self.loss_configs = loss_configs
        self.adaptive_patience = adaptive_patience
        self.adaptive_factor = adaptive_factor
        self.verbose = verbose
        
        # 当前权重
        self.current_weights = {}
        
        # 自适应调整相关
        self.best_metric = float('inf')
        self.patience_counter = 0
        self.metric_history = []
        
        # 初始化权重
        self._initialize_weights()
        
        if self.verbose:
            print("🎯 损失权重调度器初始化完成")
            self._print_schedule_info()
    
    def _initialize_weights(self):
        """初始化所有损失权重"""
        for loss_name, config in self.loss_configs.items():
            self.current_weights[loss_name] = config.get("initial_weight", 1.0)
    
    def _print_schedule_info(self):
        """打印调度信息"""
        print("\n📋 损失权重调度配置:")
        for loss_name, config in self.loss_configs.items():
            schedule_type = config.get("schedule_type", "constant")
            initial = config.get("initial_weight", 1.0)
            final = config.get("final_weight", initial)
            print(f"   {loss_name:12s}: {schedule_type:15s} {initial:.3f} → {final:.3f}")
        print()
    
    def get_current_weights(self) -> Dict[str, float]:
        """获取当前权重"""
        return self.current_weights.copy()
    
    def step(self, epoch: int, validation_metric: Optional[float] = None) -> Dict[str, float]:
        """
        更新权重
        
        Args:
            epoch: 当前epoch (1-based)
            validation_metric: 验证指标 (如RMSE，越小越好)
            
        Returns:
            updated_weights: 更新后的权重字典
        """
        # 更新各个损失权重
        for loss_name, config in self.loss_configs.items():
            new_weight = self._compute_weight(loss_name, config, epoch)
            self.current_weights[loss_name] = new_weight
        
        # 自适应调整
        if validation_metric is not None:
            self._adaptive_adjustment(validation_metric, epoch)
        
        # 打印权重变化
        if self.verbose and epoch % 10 == 0:
            self._print_current_weights(epoch)
        
        return self.get_current_weights()
    
    def _compute_weight(self, loss_name: str, config: Dict, epoch: int) -> float:
        """计算特定损失的权重"""
        schedule_type = config.get("schedule_type", "constant")
        initial_weight = config.get("initial_weight", 1.0)
        final_weight = config.get("final_weight", initial_weight)
        
        # 检查是否有开始epoch限制
        start_epoch = config.get("start_epoch", 1)
        if epoch < start_epoch:
            return 0.0
        
        # 调整epoch为相对于start_epoch的值
        relative_epoch = epoch - start_epoch + 1
        relative_total = self.total_epochs - start_epoch + 1
        
        if schedule_type == ScheduleType.CONSTANT.value:
            return initial_weight
            
        elif schedule_type == ScheduleType.LINEAR.value:
            progress = min(relative_epoch / relative_total, 1.0)
            return initial_weight + (final_weight - initial_weight) * progress
            
        elif schedule_type == ScheduleType.COSINE.value:
            progress = min(relative_epoch / relative_total, 1.0)
            cosine_progress = 0.5 * (1 + math.cos(math.pi * progress))
            return final_weight + (initial_weight - final_weight) * cosine_progress
            
        elif schedule_type == ScheduleType.EXPONENTIAL.value:
            decay_rate = config.get("decay_rate", 0.95)
            return initial_weight * (decay_rate ** (relative_epoch - 1))
            
        elif schedule_type == ScheduleType.STEP.value:
            step_size = config.get("step_size", 50)
            step_gamma = config.get("step_gamma", 0.5)
            steps = (relative_epoch - 1) // step_size
            return initial_weight * (step_gamma ** steps)
            
        elif schedule_type == ScheduleType.WARMUP_COSINE.value:
            warmup_epochs = config.get("warmup_epochs", 10)
            
            if relative_epoch <= warmup_epochs:
                # Warmup阶段：线性增长
                progress = relative_epoch / warmup_epochs
                return initial_weight * progress
            else:
                # Cosine阶段
                cosine_epochs = relative_total - warmup_epochs
                cosine_progress = (relative_epoch - warmup_epochs) / cosine_epochs
                cosine_progress = min(cosine_progress, 1.0)
                cosine_factor = 0.5 * (1 + math.cos(math.pi * cosine_progress))
                return final_weight + (initial_weight - final_weight) * cosine_factor
                
        else:
            return initial_weight
    
    def _adaptive_adjustment(self, validation_metric: float, epoch: int):
        """基于验证指标的自适应调整"""
        self.metric_history.append(validation_metric)
        
        # 检查是否有改善
        if validation_metric < self.best_metric:
            self.best_metric = validation_metric
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # 如果连续多个epoch没有改善，调整权重
        if self.patience_counter >= self.adaptive_patience:
            self._perform_adaptive_adjustment(epoch)
            self.patience_counter = 0
    
    def _perform_adaptive_adjustment(self, epoch: int):
        """执行自适应权重调整"""
        if self.verbose:
            print(f"\n🔄 Epoch {epoch}: 执行自适应权重调整")
        
        # 策略1: 增强边缘和梯度损失（通常有助于细节）
        if "edge" in self.current_weights:
            old_weight = self.current_weights["edge"]
            self.current_weights["edge"] = min(old_weight * 1.2, 2.0)
            if self.verbose:
                print(f"   edge权重: {old_weight:.3f} → {self.current_weights['edge']:.3f}")
        
        if "gradient" in self.current_weights:
            old_weight = self.current_weights["gradient"]
            self.current_weights["gradient"] = min(old_weight * 1.1, 1.0)
            if self.verbose:
                print(f"   gradient权重: {old_weight:.3f} → {self.current_weights['gradient']:.3f}")
        
        # 策略2: 适度降低感知损失（避免过拟合）
        if "perceptual" in self.current_weights:
            old_weight = self.current_weights["perceptual"]
            self.current_weights["perceptual"] = max(old_weight * self.adaptive_factor, 0.05)
            if self.verbose:
                print(f"   perceptual权重: {old_weight:.3f} → {self.current_weights['perceptual']:.3f}")
    
    def _print_current_weights(self, epoch: int):
        """打印当前权重"""
        print(f"\n📊 Epoch {epoch} 当前损失权重:")
        for loss_name, weight in self.current_weights.items():
            print(f"   {loss_name:12s}: {weight:.4f}")
    
    def get_schedule_summary(self) -> str:
        """获取调度器摘要信息"""
        summary = "Loss Weight Scheduler Summary:\n"
        summary += f"Total Epochs: {self.total_epochs}\n"
        summary += f"Adaptive Patience: {self.adaptive_patience}\n"
        summary += "Loss Configurations:\n"
        
        for loss_name, config in self.loss_configs.items():
            schedule_type = config.get("schedule_type", "constant")
            initial = config.get("initial_weight", 1.0)
            final = config.get("final_weight", initial)
            summary += f"  {loss_name}: {schedule_type} ({initial:.3f} → {final:.3f})\n"
        
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


class PresetLossSchedulers:
    """预设的损失调度器配置"""
    
    @staticmethod
    def get_shadow_removal_config(total_epochs: int) -> Dict[str, Dict]:
        """
        阴影去除任务的推荐配置
        
        策略：
        1. MSE保持恒定作为基础
        2. SSIM逐渐增强，保持结构
        3. Edge权重先增后减，重点处理边缘
        4. Gradient逐渐增强，确保平滑
        5. Boundary在中期达到峰值
        6. Perceptual和Histogram保持适中
        """
        return {
            "mse": {
                "schedule_type": "constant",
                "initial_weight": 1.0,
                "final_weight": 1.0
            },
            "ssim": {
                "schedule_type": "warmup_cosine",
                "initial_weight": 0.1,
                "final_weight": 0.4,
                "warmup_epochs": max(10, total_epochs // 20)
            },
            "edge": {
                "schedule_type": "cosine",
                "initial_weight": 0.3,
                "final_weight": 0.8,
                "start_epoch": 5
            },
            "gradient": {
                "schedule_type": "linear",
                "initial_weight": 0.2,
                "final_weight": 0.5
            },
            "boundary": {
                "schedule_type": "warmup_cosine",
                "initial_weight": 0.3,
                "final_weight": 0.6,
                "warmup_epochs": max(15, total_epochs // 15)
            },
            "transparency": {
                "schedule_type": "linear",
                "initial_weight": 0.1,
                "final_weight": 0.3,
                "start_epoch": 10
            },
            "perceptual": {
                "schedule_type": "constant",
                "initial_weight": 0.2,
                "final_weight": 0.2
            },
            "histogram": {
                "schedule_type": "linear",
                "initial_weight": 0.1,
                "final_weight": 0.25,
                "start_epoch": 20
            }
        }
    
    @staticmethod
    def get_fine_tuning_config(total_epochs: int) -> Dict[str, Dict]:
        """
        微调阶段的配置 - 更注重细节优化
        """
        return {
            "mse": {
                "schedule_type": "exponential",
                "initial_weight": 1.0,
                "decay_rate": 0.98
            },
            "ssim": {
                "schedule_type": "linear",
                "initial_weight": 0.3,
                "final_weight": 0.6
            },
            "edge": {
                "schedule_type": "linear",
                "initial_weight": 0.5,
                "final_weight": 1.0
            },
            "gradient": {
                "schedule_type": "linear",
                "initial_weight": 0.3,
                "final_weight": 0.7
            },
            "boundary": {
                "schedule_type": "constant",
                "initial_weight": 0.8,
                "final_weight": 0.8
            },
            "transparency": {
                "schedule_type": "linear",
                "initial_weight": 0.2,
                "final_weight": 0.4
            },
            "perceptual": {
                "schedule_type": "exponential",
                "initial_weight": 0.3,
                "decay_rate": 0.95
            },
            "histogram": {
                "schedule_type": "linear",
                "initial_weight": 0.2,
                "final_weight": 0.3
            }
        }
    
    @staticmethod
    def get_aggressive_config(total_epochs: int) -> Dict[str, Dict]:
        """
        激进配置 - 快速收敛，适合数据充足的情况
        """
        return {
            "mse": {
                "schedule_type": "constant",
                "initial_weight": 1.0,
                "final_weight": 1.0
            },
            "ssim": {
                "schedule_type": "linear",
                "initial_weight": 0.5,
                "final_weight": 0.8
            },
            "edge": {
                "schedule_type": "linear",
                "initial_weight": 0.8,
                "final_weight": 1.2
            },
            "gradient": {
                "schedule_type": "linear",
                "initial_weight": 0.4,
                "final_weight": 0.8
            },
            "boundary": {
                "schedule_type": "linear",
                "initial_weight": 0.6,
                "final_weight": 1.0
            },
            "transparency": {
                "schedule_type": "linear",
                "initial_weight": 0.3,
                "final_weight": 0.5
            },
            "perceptual": {
                "schedule_type": "constant",
                "initial_weight": 0.3,
                "final_weight": 0.3
            },
            "histogram": {
                "schedule_type": "linear",
                "initial_weight": 0.2,
                "final_weight": 0.4
            }
        }


def create_loss_scheduler_from_config(config, total_epochs: int) -> LossWeightScheduler:
    """
    从配置文件创建损失调度器
    
    Args:
        config: 配置对象，包含LOSS_SCHEDULER部分
        total_epochs: 总训练轮数
        
    Returns:
        LossWeightScheduler实例
    """
    try:
        # 获取调度器配置
        scheduler_config = config.LOSS_SCHEDULER
        
        # 获取预设类型
        preset_type = getattr(scheduler_config, 'PRESET', 'shadow_removal')
        
        # 根据预设类型获取配置
        if preset_type == 'shadow_removal':
            loss_configs = PresetLossSchedulers.get_shadow_removal_config(total_epochs)
        elif preset_type == 'fine_tuning':
            loss_configs = PresetLossSchedulers.get_fine_tuning_config(total_epochs)
        elif preset_type == 'aggressive':
            loss_configs = PresetLossSchedulers.get_aggressive_config(total_epochs)
        else:
            # 使用默认配置
            print(f"⚠️  未知的预设类型 '{preset_type}'，使用默认 shadow_removal 配置")
            loss_configs = PresetLossSchedulers.get_shadow_removal_config(total_epochs)
        
        # 暂时跳过 OVERRIDES 处理，避免配置错误
        # TODO: 后续可以添加更完善的覆盖机制
        
        # 创建调度器
        scheduler = LossWeightScheduler(
            total_epochs=total_epochs,
            loss_configs=loss_configs,
            adaptive_patience=getattr(scheduler_config, 'ADAPTIVE_PATIENCE', 5),
            adaptive_factor=getattr(scheduler_config, 'ADAPTIVE_FACTOR', 0.8),
            verbose=getattr(scheduler_config, 'VERBOSE', True)
        )
        
        return scheduler
        
    except Exception as e:
        print(f"❌ 创建损失调度器失败: {e}")
        # 返回一个简单的默认调度器
        loss_configs = PresetLossSchedulers.get_shadow_removal_config(total_epochs)
        return LossWeightScheduler(
            total_epochs=total_epochs,
            loss_configs=loss_configs,
            adaptive_patience=5,
            adaptive_factor=0.8,
            verbose=True
        )


if __name__ == '__main__':
    # 测试损失调度器
    print("🧪 测试损失权重调度器")
    
    # 创建测试配置
    loss_configs = PresetLossSchedulers.get_shadow_removal_config(100)
    
    # 创建调度器
    scheduler = LossWeightScheduler(
        total_epochs=100,
        loss_configs=loss_configs,
        verbose=True
    )
    
    # 模拟训练过程
    print("\n🚀 模拟训练过程:")
    test_epochs = [1, 10, 25, 50, 75, 100]
    test_metrics = [0.5, 0.3, 0.25, 0.22, 0.21, 0.20]
    
    for epoch, metric in zip(test_epochs, test_metrics):
        weights = scheduler.step(epoch, metric)
        print(f"\nEpoch {epoch:3d} (RMSE: {metric:.3f}):")
        for loss_name, weight in weights.items():
            print(f"  {loss_name:12s}: {weight:.4f}")