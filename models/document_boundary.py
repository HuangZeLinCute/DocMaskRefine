"""
文档边界注意力模块 - 专门处理文档阴影去除中的边界问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DocumentBoundaryAttention(nn.Module):
    """
    文档感知的边界注意力模块
    专门为文档阴影去除任务设计，解决黑边问题
    """
    
    def __init__(self, in_channels, reduction=16):
        super(DocumentBoundaryAttention, self).__init__()
        self.in_channels = in_channels
        
        # 1. 边界检测分支
        self.edge_detector = EdgeDetectionBranch(in_channels)
        
        # 2. 文档区域分析分支
        self.document_analyzer = DocumentRegionBranch(in_channels)
        
        # 3. 边界平滑分支
        self.boundary_smoother = BoundarySmoothingBranch(in_channels, reduction)
        
        # 4. 特征融合
        self.feature_fusion = FeatureFusionModule(in_channels)
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 输入特征
        Returns:
            refined_x: [B, C, H, W] 边界优化后的特征
        """
        # 1. 检测边界区域
        edge_map = self.edge_detector(x)
        
        # 2. 分析文档区域（文字 vs 背景）
        region_map = self.document_analyzer(x)
        
        # 3. 生成边界平滑权重
        smooth_weights = self.boundary_smoother(x, edge_map, region_map)
        
        # 4. 应用边界优化
        refined_x = self.feature_fusion(x, smooth_weights, edge_map, region_map)
        
        return refined_x


class EdgeDetectionBranch(nn.Module):
    """边界检测分支 - 检测阴影边界"""
    
    def __init__(self, in_channels):
        super(EdgeDetectionBranch, self).__init__()
        
        # Sobel算子用于边缘检测
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
        
        # 学习的边缘检测器
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        检测边界区域
        Returns:
            edge_map: [B, 1, H, W] 边界概率图
        """
        # 使用学习的边缘检测器
        edge_map = self.edge_conv(x)
        
        # 也可以结合传统Sobel算子（可选）
        # gray = x.mean(dim=1, keepdim=True)  # 转灰度
        # sobel_x = F.conv2d(gray, self.sobel_x.to(x.device), padding=1)
        # sobel_y = F.conv2d(gray, self.sobel_y.to(x.device), padding=1)
        # sobel_edge = torch.sqrt(sobel_x ** 2 + sobel_y ** 2)
        # edge_map = edge_map + 0.3 * sobel_edge  # 结合传统和学习的边缘
        
        return edge_map


class DocumentRegionBranch(nn.Module):
    """文档区域分析分支 - 区分文字区域和背景区域"""
    
    def __init__(self, in_channels):
        super(DocumentRegionBranch, self).__init__()
        
        # 文档区域分析网络
        self.region_analyzer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 2, 1),  # 2通道：文字概率 + 背景概率
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        """
        分析文档区域
        Returns:
            region_map: [B, 2, H, W] 区域概率图 (文字, 背景)
        """
        region_map = self.region_analyzer(x)
        return region_map


class BoundarySmoothingBranch(nn.Module):
    """边界平滑分支 - 生成边界平滑权重"""
    
    def __init__(self, in_channels, reduction=16):
        super(BoundarySmoothingBranch, self).__init__()
        
        mid_channels = max(8, in_channels // reduction)
        
        # 边界平滑权重生成器
        self.smooth_generator = nn.Sequential(
            nn.Conv2d(in_channels + 1 + 2, mid_channels, 3, padding=1),  # +1(edge) +2(region)
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1),
            nn.Sigmoid()
        )
        
        # 高斯模糊核用于边界平滑
        self.gaussian_blur = self._create_gaussian_kernel(kernel_size=5, sigma=1.0)
        
    def _create_gaussian_kernel(self, kernel_size=5, sigma=1.0):
        """创建高斯模糊核"""
        x = torch.arange(kernel_size, dtype=torch.float32)
        x = x - kernel_size // 2
        gauss_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        
        gauss_2d = gauss_1d.view(1, -1) * gauss_1d.view(-1, 1)
        gauss_2d = gauss_2d.view(1, 1, kernel_size, kernel_size)
        
        return gauss_2d
        
    def forward(self, x, edge_map, region_map):
        """
        生成边界平滑权重
        Args:
            x: [B, C, H, W] 输入特征
            edge_map: [B, 1, H, W] 边界图
            region_map: [B, 2, H, W] 区域图
        Returns:
            smooth_weights: [B, C, H, W] 平滑权重
        """
        # 拼接所有信息
        combined = torch.cat([x, edge_map, region_map], dim=1)
        
        # 生成平滑权重
        smooth_weights = self.smooth_generator(combined)
        
        return smooth_weights


class FeatureFusionModule(nn.Module):
    """特征融合模块 - 应用边界优化"""
    
    def __init__(self, in_channels):
        super(FeatureFusionModule, self).__init__()
        
        # 自适应融合权重
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1)
        )
        
    def forward(self, x, smooth_weights, edge_map, region_map):
        """
        应用边界优化
        Args:
            x: [B, C, H, W] 原始特征
            smooth_weights: [B, C, H, W] 平滑权重
            edge_map: [B, 1, H, W] 边界图
            region_map: [B, 2, H, W] 区域图
        Returns:
            refined_x: [B, C, H, W] 优化后的特征
        """
        # 在边界区域应用平滑
        # 背景区域（region_map[:, 1:2]）应用更强的平滑
        background_mask = region_map[:, 1:2]  # [B, 1, H, W] 背景概率
        
        # 在边界+背景区域应用平滑
        boundary_background = edge_map * background_mask
        
        # 生成平滑后的特征
        smoothed_x = x * smooth_weights
        
        # 自适应融合原始特征和平滑特征
        # 在边界背景区域更多使用平滑特征，在文字区域保持原始特征
        fusion_weight = boundary_background.expand_as(x)
        blended_x = x * (1 - fusion_weight) + smoothed_x * fusion_weight
        
        # 最终融合
        fused_features = torch.cat([x, blended_x], dim=1)
        refined_x = self.fusion_conv(fused_features)
        
        return refined_x


# 测试代码
if __name__ == '__main__':
    # 测试文档边界注意力模块
    x = torch.randn(2, 64, 256, 256)
    
    doc_boundary = DocumentBoundaryAttention(in_channels=64)
    
    refined_x = doc_boundary(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {refined_x.shape}")
    print("文档边界注意力模块测试通过！")