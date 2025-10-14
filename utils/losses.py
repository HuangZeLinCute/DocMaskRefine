import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure
import torchvision.models as models



class ReconstructionLoss(nn.Module):
    """
    简化的图像重建损失函数 - 只包含MSE和SSIM
    包含：
    - MSE Loss (基础重建)
    - SSIM Loss (结构相似性)

    支持动态权重调整
    """
    def __init__(self, 
                 mse_weight: float = 1.0,
                 ssim_weight: float = 0.2,
                 fade_width: int = 10,
                 min_alpha: float = 0.3):
        """
        Args:
            mse_weight: MSE损失权重
            ssim_weight: SSIM损失权重
        """
        super(ReconstructionLoss, self).__init__()
        self.mse = nn.MSELoss()
        
        # 初始权重 - 只使用MSE和SSIM
        self.weights = {
            'mse': mse_weight,
            'ssim': ssim_weight
        }
    
    def update_weights(self, new_weights: dict):
        """
        更新损失权重
        
        Args:
            new_weights: 新的权重字典，例如 {'ssim': 0.3, 'edge': 0.5}
        """
        for loss_name, weight in new_weights.items():
            if loss_name in self.weights:
                self.weights[loss_name] = weight
    
    def get_weights(self) -> dict:
        """获取当前权重"""
        return self.weights.copy()

    def forward(self, pred, target, shadow_mask=None):
        """
        Args:
            pred: [B, 3, H, W] 预测图像
            target: [B, 3, H, W] 目标图像
            shadow_mask: [B, 1, H, W] 阴影掩码 (可选)
        """
        # 基础损失
        loss_mse = self.mse(pred, target)
        
        # 结构相似度损失
        loss_ssim = 1 - structural_similarity_index_measure(pred, target, data_range=1.0)
        
        # 使用动态权重计算总损失
        total_loss = (self.weights['mse'] * loss_mse + 
                     self.weights['ssim'] * loss_ssim)

        return total_loss, loss_mse, loss_ssim


if __name__ == '__main__':
    # 假设输入是 batch=2, 3通道RGB图, 256x256
    pred = torch.rand((2, 3, 512, 512))  # 模型输出
    target = torch.rand((2, 3, 512, 512))  # GT图像

    # 定义Loss
    criterion = ReconstructionLoss(ssim_weight=0.2)

    # 前向计算
    loss, loss_mse, loss_ssim = criterion(pred, target)

    print("Total Loss:", loss.item())
    print("MSE Loss:", loss_mse.item())
    print("SSIM Loss:", loss_ssim.item())
