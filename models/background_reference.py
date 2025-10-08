"""
背景参考引导模块 - 解决阴影区域与背景不一致的问题
让阴影修复后能贴近整体背景图
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BackgroundReferenceModule(nn.Module):
    """
    背景参考引导模块 - 平衡版本
    从非阴影区域提取背景特征，然后应用到阴影区域
    确保阴影修复后与整体背景一致，同时不干扰边缘处理
    """
    
    def __init__(self, in_channels, reference_radius=16, strength=0.6):
        super(BackgroundReferenceModule, self).__init__()
        self.reference_radius = reference_radius
        self.strength = strength  # 控制背景参考的强度
        
        # 背景特征提取器
        self.background_extractor = BackgroundFeatureExtractor(in_channels)
        
        # 阴影区域检测器
        self.shadow_detector = ShadowRegionDetector(in_channels)
        
        # 背景特征传播器
        self.feature_propagator = BackgroundFeaturePropagator(in_channels)
        
        # 自适应融合器
        self.adaptive_fusion = AdaptiveBackgroundFusion(in_channels)
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 输入特征
        Returns:
            enhanced_x: [B, C, H, W] 背景一致性增强后的特征
        """
        # 1. 检测阴影区域
        shadow_mask = self.shadow_detector(x)
        
        # 2. 提取背景特征
        background_features = self.background_extractor(x, shadow_mask)
        
        # 3. 将背景特征传播到阴影区域
        propagated_features = self.feature_propagator(x, background_features, shadow_mask)
        
        # 4. 自适应融合原始特征和背景引导特征
        enhanced_x = self.adaptive_fusion(x, propagated_features, shadow_mask)
        
        return enhanced_x


class BackgroundFeatureExtractor(nn.Module):
    """背景特征提取器 - 从非阴影区域提取背景特征"""
    
    def __init__(self, in_channels):
        super(BackgroundFeatureExtractor, self).__init__()
        
        # 多尺度背景特征提取
        self.multi_scale_extractors = nn.ModuleList([
            self._make_extractor(in_channels, scale=1),  # 原始尺度
            self._make_extractor(in_channels, scale=2),  # 1/2尺度
            self._make_extractor(in_channels, scale=4),  # 1/4尺度
        ])
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def _make_extractor(self, in_channels, scale):
        """创建指定尺度的特征提取器"""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, shadow_mask):
        """
        从非阴影区域提取背景特征
        Args:
            x: [B, C, H, W] 输入特征
            shadow_mask: [B, 1, H, W] 阴影掩码 (1=阴影, 0=背景)
        Returns:
            background_features: [B, C, H, W] 背景特征
        """
        # 创建背景掩码 (非阴影区域)
        background_mask = 1.0 - shadow_mask
        
        # 多尺度特征提取
        multi_scale_features = []
        
        for i, extractor in enumerate(self.multi_scale_extractors):
            scale = 2 ** i
            
            if scale > 1:
                # 下采样
                x_scaled = F.avg_pool2d(x, scale)
                mask_scaled = F.avg_pool2d(background_mask, scale)
            else:
                x_scaled = x
                mask_scaled = background_mask
            
            # 提取特征
            features = extractor(x_scaled)
            
            # 只保留背景区域的特征
            features = features * mask_scaled
            
            # 上采样回原始尺寸
            if scale > 1:
                features = F.interpolate(features, size=x.shape[2:], mode='bilinear', align_corners=False)
            
            multi_scale_features.append(features)
        
        # 融合多尺度特征
        combined_features = torch.cat(multi_scale_features, dim=1)
        background_features = self.feature_fusion(combined_features)
        
        return background_features


class ShadowRegionDetector(nn.Module):
    """阴影区域检测器 - 自动检测阴影区域"""
    
    def __init__(self, in_channels):
        super(ShadowRegionDetector, self).__init__()
        
        # 阴影检测网络
        self.shadow_detector = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # 形态学操作用于平滑掩码
        self.morphology_kernel = self._create_morphology_kernel()
        
    def _create_morphology_kernel(self, kernel_size=5):
        """创建形态学操作的核"""
        kernel = torch.ones(1, 1, kernel_size, kernel_size)
        return kernel
        
    def forward(self, x):
        """
        检测阴影区域
        Args:
            x: [B, C, H, W] 输入特征
        Returns:
            shadow_mask: [B, 1, H, W] 阴影掩码
        """
        # 检测阴影
        shadow_mask = self.shadow_detector(x)
        
        # 形态学平滑 (可选)
        kernel = self.morphology_kernel.to(x.device)
        shadow_mask = F.conv2d(shadow_mask, kernel, padding=2)
        shadow_mask = torch.clamp(shadow_mask / (kernel.sum()), 0, 1)
        
        return shadow_mask


class BackgroundFeaturePropagator(nn.Module):
    """背景特征传播器 - 将背景特征传播到阴影区域"""
    
    def __init__(self, in_channels, propagation_steps=3):
        super(BackgroundFeaturePropagator, self).__init__()
        self.propagation_steps = propagation_steps
        
        # 特征传播网络
        self.propagation_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 传播权重生成器
        self.weight_generator = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, background_features, shadow_mask):
        """
        将背景特征传播到阴影区域
        Args:
            x: [B, C, H, W] 原始特征
            background_features: [B, C, H, W] 背景特征
            shadow_mask: [B, 1, H, W] 阴影掩码
        Returns:
            propagated_features: [B, C, H, W] 传播后的特征
        """
        current_features = x
        
        for step in range(self.propagation_steps):
            # 结合当前特征和背景特征
            combined = torch.cat([current_features, background_features], dim=1)
            
            # 传播特征
            propagated = self.propagation_conv(combined)
            
            # 生成传播权重
            weight_input = torch.cat([current_features, shadow_mask], dim=1)
            propagation_weight = self.weight_generator(weight_input)
            
            # 在阴影区域应用传播
            current_features = current_features * (1 - shadow_mask * propagation_weight) + \
                              propagated * (shadow_mask * propagation_weight)
        
        return current_features


class AdaptiveBackgroundFusion(nn.Module):
    """自适应背景融合器 - 智能融合原始特征和背景引导特征"""
    
    def __init__(self, in_channels):
        super(AdaptiveBackgroundFusion, self).__init__()
        
        # 融合权重生成器
        self.fusion_weight_generator = nn.Sequential(
            nn.Conv2d(in_channels * 2 + 1, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, 1),
            nn.Sigmoid()
        )
        
        # 最终特征优化
        self.feature_refiner = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1)
        )
        
    def forward(self, original_features, propagated_features, shadow_mask):
        """
        自适应融合特征
        Args:
            original_features: [B, C, H, W] 原始特征
            propagated_features: [B, C, H, W] 背景传播特征
            shadow_mask: [B, 1, H, W] 阴影掩码
        Returns:
            fused_features: [B, C, H, W] 融合后的特征
        """
        # 生成自适应融合权重
        fusion_input = torch.cat([original_features, propagated_features, shadow_mask], dim=1)
        fusion_weights = self.fusion_weight_generator(fusion_input)
        
        # 自适应融合
        # 在阴影区域更多使用背景引导特征，在非阴影区域保持原始特征
        shadow_weight = shadow_mask.expand_as(fusion_weights)
        adaptive_weights = fusion_weights * shadow_weight
        
        fused_features = original_features * (1 - adaptive_weights) + \
                        propagated_features * adaptive_weights
        
        # 特征优化
        refined_features = self.feature_refiner(fused_features)
        
        # 残差连接
        final_features = original_features + refined_features
        
        return final_features


# 测试代码
if __name__ == '__main__':
    # 测试背景参考引导模块
    x = torch.randn(2, 64, 128, 128)
    
    bg_ref_module = BackgroundReferenceModule(in_channels=64)
    
    output = bg_ref_module(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print("背景参考引导模块测试通过！")