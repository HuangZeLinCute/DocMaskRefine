"""
黑边抑制模块 - 专门针对阴影边界的黑边问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BlackEdgeSuppressor(nn.Module):
    """
    简化的黑边抑制模块
    专门检测和修复阴影边界的黑边问题
    """
    
    def __init__(self, in_channels, suppression_strength=0.5):
        super(BlackEdgeSuppressor, self).__init__()
        self.suppression_strength = suppression_strength
        
        # 简化的黑边检测和修复
        self.edge_smoother = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 2, 5, padding=2),  # 大核平滑
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, 3, padding=1),
            nn.Sigmoid()  # 输出权重
        )
        
        # 残差连接权重
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 输入特征
        Returns:
            corrected_x: [B, C, H, W] 黑边修复后的特征
        """
        # 生成平滑权重
        smooth_weights = self.edge_smoother(x)
        
        # 应用平滑（简单的均值滤波）
        smoothed = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        
        # 加权融合
        output = x * (1 - smooth_weights * self.suppression_strength) + \
                smoothed * (smooth_weights * self.suppression_strength)
        
        # 残差连接
        output = x + self.residual_weight * (output - x)
        
        return output


class BlackEdgeDetector(nn.Module):
    """黑边检测器 - 检测可能的黑边区域"""
    
    def __init__(self, in_channels):
        super(BlackEdgeDetector, self).__init__()
        
        # 多尺度边缘检测
        self.edge_detectors = nn.ModuleList([
            self._make_edge_detector(in_channels, kernel_size=3),
            self._make_edge_detector(in_channels, kernel_size=5),
            self._make_edge_detector(in_channels, kernel_size=7),
        ])
        
        # 黑边特征分析器
        self.black_analyzer = nn.Sequential(
            nn.Conv2d(in_channels + 3, in_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
    def _make_edge_detector(self, in_channels, kernel_size):
        """创建边缘检测器"""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size, padding=kernel_size//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        检测黑边区域
        Returns:
            black_edge_mask: [B, 1, H, W] 黑边概率图
        """
        # 多尺度边缘检测
        edge_maps = []
        for detector in self.edge_detectors:
            edge_map = detector(x)
            edge_maps.append(edge_map)
        
        # 融合多尺度边缘信息
        combined_edges = torch.cat(edge_maps, dim=1)  # [B, 3, H, W]
        
        # 结合原始特征分析黑边
        combined_features = torch.cat([x, combined_edges], dim=1)
        black_edge_mask = self.black_analyzer(combined_features)
        
        return black_edge_mask


class BlackEdgeCorrector(nn.Module):
    """黑边修复器 - 修复检测到的黑边区域"""
    
    def __init__(self, in_channels):
        super(BlackEdgeCorrector, self).__init__()
        
        # 上下文特征提取器
        self.context_extractor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 5, padding=2),  # 大感受野
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 5, padding=2),
            nn.ReLU(inplace=True)
        )
        
        # 黑边修复网络
        self.corrector = nn.Sequential(
            nn.Conv2d(in_channels * 2 + 1, in_channels * 2, 3, padding=1),  # +1 for mask
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1)
        )
        
        # 平滑滤波器
        self.smoother = self._create_gaussian_filter()
        
    def _create_gaussian_filter(self):
        """创建高斯平滑滤波器"""
        kernel_size = 5
        sigma = 1.0
        
        x = torch.arange(kernel_size, dtype=torch.float32)
        x = x - kernel_size // 2
        gauss_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        
        gauss_2d = gauss_1d.view(1, -1) * gauss_1d.view(-1, 1)
        gauss_2d = gauss_2d.view(1, 1, kernel_size, kernel_size)
        
        return gauss_2d
        
    def forward(self, x, black_edge_mask):
        """
        修复黑边区域
        Args:
            x: [B, C, H, W] 输入特征
            black_edge_mask: [B, 1, H, W] 黑边掩码
        Returns:
            corrected_x: [B, C, H, W] 修复后的特征
        """
        # 提取上下文特征
        context_features = self.context_extractor(x)
        
        # 应用平滑滤波到黑边区域
        # 确保高斯核在正确的设备上
        if not hasattr(self, '_smoother_device') or self._smoother_device != x.device:
            self.smoother = self.smoother.to(x.device)
            self._smoother_device = x.device
            
        # 对所有通道一次性应用高斯滤波
        smoothed_x = F.conv2d(x, self.smoother.expand(x.size(1), -1, -1, -1), 
                             padding=2, groups=x.size(1))
        
        # 在黑边区域混合平滑特征
        blended_features = x * (1 - black_edge_mask) + smoothed_x * black_edge_mask
        
        # 结合所有信息进行修复
        combined = torch.cat([x, blended_features, black_edge_mask], dim=1)
        correction = self.corrector(combined)
        
        # 残差连接
        corrected_x = x + correction
        
        return corrected_x


class AdaptiveFusion(nn.Module):
    """自适应融合模块"""
    
    def __init__(self, in_channels):
        super(AdaptiveFusion, self).__init__()
        
        self.fusion_weights = nn.Sequential(
            nn.Conv2d(in_channels * 2 + 1, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, original, corrected, mask):
        """
        自适应融合原始和修复后的特征
        """
        # 生成融合权重
        combined = torch.cat([original, corrected, mask], dim=1)
        weights = self.fusion_weights(combined)
        
        # 自适应融合
        output = original * (1 - weights) + corrected * weights
        
        return output


# 测试代码
if __name__ == '__main__':
    # 测试黑边抑制模块
    x = torch.randn(2, 64, 128, 128)
    
    suppressor = BlackEdgeSuppressor(in_channels=64)
    
    output = suppressor(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print("黑边抑制模块测试通过！")