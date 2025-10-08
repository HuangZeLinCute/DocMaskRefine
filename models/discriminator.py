"""
轻量级文档质量判别器 - 专门用于文档阴影去除任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DocumentQualityDiscriminator(nn.Module):
    """
    轻量级文档质量判别器
    专门判断文档图像的边界质量、文字清晰度和背景一致性
    """
    
    def __init__(self, input_channels=3, base_channels=32):
        super(DocumentQualityDiscriminator, self).__init__()
        
        # 多尺度特征提取
        self.multi_scale_extractor = MultiScaleFeatureExtractor(input_channels, base_channels)
        
        # 边界质量分析器
        self.boundary_analyzer = BoundaryQualityAnalyzer(base_channels * 4)
        
        # 文档内容分析器
        self.content_analyzer = DocumentContentAnalyzer(base_channels * 4)
        
        # 最终判别器
        self.final_discriminator = FinalDiscriminator(base_channels * 8)
        
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] 输入图像
        Returns:
            validity: [B, 1] 图像质量分数 (0-1, 1表示高质量)
            boundary_score: [B, 1] 边界质量分数
            content_score: [B, 1] 内容质量分数
        """
        # 多尺度特征提取
        multi_scale_features = self.multi_scale_extractor(x)
        
        # 边界质量分析
        boundary_features, boundary_score = self.boundary_analyzer(multi_scale_features)
        
        # 文档内容分析
        content_features, content_score = self.content_analyzer(multi_scale_features)
        
        # 融合特征
        combined_features = torch.cat([boundary_features, content_features], dim=1)
        
        # 最终判别
        validity = self.final_discriminator(combined_features)
        
        return validity, boundary_score, content_score


class MultiScaleFeatureExtractor(nn.Module):
    """多尺度特征提取器 - 提取不同尺度的文档特征"""
    
    def __init__(self, input_channels=3, base_channels=32):
        super(MultiScaleFeatureExtractor, self).__init__()
        
        # 第一层：保持原始分辨率
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 第二层：1/2分辨率
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 第三层：1/4分辨率
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] 输入图像
        Returns:
            features: [B, base_channels*4, H/4, W/4] 多尺度融合特征
        """
        # 逐层提取特征
        f1 = self.conv1(x)          # [B, base_channels, H, W]
        f2 = self.conv2(f1)         # [B, base_channels*2, H/2, W/2]
        f3 = self.conv3(f2)         # [B, base_channels*4, H/4, W/4]
        
        # 融合特征
        features = self.feature_fusion(f3)
        
        return features


class BoundaryQualityAnalyzer(nn.Module):
    """边界质量分析器 - 专门分析阴影边界的质量"""
    
    def __init__(self, input_channels):
        super(BoundaryQualityAnalyzer, self).__init__()
        
        # 边界检测分支
        self.edge_detector = nn.Sequential(
            nn.Conv2d(input_channels, input_channels // 2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(input_channels // 2, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        
        # 边界质量评估
        self.boundary_evaluator = nn.Sequential(
            nn.Conv2d(input_channels + 1, input_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(8),
            nn.Conv2d(input_channels, input_channels // 2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(input_channels // 2, 1),
            nn.Sigmoid()
        )
        
        # 特征输出
        self.feature_output = nn.Sequential(
            nn.Conv2d(input_channels + 1, input_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(input_channels, input_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self, features):
        """
        Args:
            features: [B, C, H, W] 输入特征
        Returns:
            boundary_features: [B, C, 4, 4] 边界特征
            boundary_score: [B, 1] 边界质量分数
        """
        # 检测边界
        edge_map = self.edge_detector(features)
        
        # 结合边界信息
        combined = torch.cat([features, edge_map], dim=1)
        
        # 评估边界质量
        boundary_score = self.boundary_evaluator(combined)
        
        # 输出特征
        boundary_features = self.feature_output(combined)
        
        return boundary_features, boundary_score


class DocumentContentAnalyzer(nn.Module):
    """文档内容分析器 - 分析文字清晰度和背景一致性"""
    
    def __init__(self, input_channels):
        super(DocumentContentAnalyzer, self).__init__()
        
        # 文字区域检测
        self.text_detector = nn.Sequential(
            nn.Conv2d(input_channels, input_channels // 2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(input_channels // 2, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        
        # 背景一致性分析
        self.background_analyzer = nn.Sequential(
            nn.Conv2d(input_channels, input_channels // 2, 5, 1, 2),  # 更大的感受野
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(input_channels // 2, input_channels // 4, 5, 1, 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 内容质量评估
        self.content_evaluator = nn.Sequential(
            nn.Conv2d(input_channels + input_channels // 4 + 1, input_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(8),
            nn.Conv2d(input_channels, input_channels // 2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(input_channels // 2, 1),
            nn.Sigmoid()
        )
        
        # 特征输出
        self.feature_output = nn.Sequential(
            nn.Conv2d(input_channels + input_channels // 4 + 1, input_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(input_channels, input_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self, features):
        """
        Args:
            features: [B, C, H, W] 输入特征
        Returns:
            content_features: [B, C, 4, 4] 内容特征
            content_score: [B, 1] 内容质量分数
        """
        # 检测文字区域
        text_map = self.text_detector(features)
        
        # 分析背景一致性
        background_features = self.background_analyzer(features)
        
        # 结合所有信息
        combined = torch.cat([features, background_features, text_map], dim=1)
        
        # 评估内容质量
        content_score = self.content_evaluator(combined)
        
        # 输出特征
        content_features = self.feature_output(combined)
        
        return content_features, content_score


class FinalDiscriminator(nn.Module):
    """最终判别器 - 综合判断图像质量"""
    
    def __init__(self, input_channels):
        super(FinalDiscriminator, self).__init__()
        
        self.discriminator = nn.Sequential(
            # 特征融合
            nn.Conv2d(input_channels, input_channels // 2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 全局池化
            nn.AdaptiveAvgPool2d(2),
            
            # 最终判别
            nn.Conv2d(input_channels // 2, input_channels // 4, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            
            # 全连接层
            nn.Linear(input_channels // 4, input_channels // 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(input_channels // 8, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features):
        """
        Args:
            features: [B, C, 4, 4] 融合特征
        Returns:
            validity: [B, 1] 图像质量分数
        """
        validity = self.discriminator(features)
        return validity


class DiscriminatorLoss(nn.Module):
    """判别器损失函数"""
    
    def __init__(self, boundary_weight=0.3, content_weight=0.3, adversarial_weight=0.4):
        super(DiscriminatorLoss, self).__init__()
        self.boundary_weight = boundary_weight
        self.content_weight = content_weight
        self.adversarial_weight = adversarial_weight
        
        self.bce_loss = nn.BCELoss()
        
    def forward(self, real_validity, real_boundary, real_content,
                fake_validity, fake_boundary, fake_content):
        """
        计算判别器损失
        Args:
            real_*: 真实图像的判别结果
            fake_*: 生成图像的判别结果
        Returns:
            total_loss: 总损失
            losses: 各项损失的字典
        """
        batch_size = real_validity.size(0)
        
        # 真实标签和虚假标签
        real_labels = torch.ones_like(real_validity)
        fake_labels = torch.zeros_like(fake_validity)
        
        # 对抗损失
        real_adv_loss = self.bce_loss(real_validity, real_labels)
        fake_adv_loss = self.bce_loss(fake_validity, fake_labels)
        adversarial_loss = (real_adv_loss + fake_adv_loss) / 2
        
        # 边界质量损失
        real_boundary_loss = self.bce_loss(real_boundary, real_labels)
        fake_boundary_loss = self.bce_loss(fake_boundary, fake_labels)
        boundary_loss = (real_boundary_loss + fake_boundary_loss) / 2
        
        # 内容质量损失
        real_content_loss = self.bce_loss(real_content, real_labels)
        fake_content_loss = self.bce_loss(fake_content, fake_labels)
        content_loss = (real_content_loss + fake_content_loss) / 2
        
        # 总损失
        total_loss = (self.adversarial_weight * adversarial_loss +
                     self.boundary_weight * boundary_loss +
                     self.content_weight * content_loss)
        
        losses = {
            'adversarial': adversarial_loss,
            'boundary': boundary_loss,
            'content': content_loss,
            'total': total_loss
        }
        
        return total_loss, losses


class GeneratorAdversarialLoss(nn.Module):
    """生成器对抗损失"""
    
    def __init__(self, boundary_weight=0.3, content_weight=0.3, adversarial_weight=0.4):
        super(GeneratorAdversarialLoss, self).__init__()
        self.boundary_weight = boundary_weight
        self.content_weight = content_weight
        self.adversarial_weight = adversarial_weight
        
        self.bce_loss = nn.BCELoss()
        
    def forward(self, fake_validity, fake_boundary, fake_content):
        """
        计算生成器对抗损失
        Args:
            fake_*: 生成图像的判别结果
        Returns:
            total_loss: 总对抗损失
        """
        batch_size = fake_validity.size(0)
        
        # 生成器希望判别器认为生成的图像是真实的
        real_labels = torch.ones_like(fake_validity)
        
        # 对抗损失
        adversarial_loss = self.bce_loss(fake_validity, real_labels)
        
        # 边界质量损失
        boundary_loss = self.bce_loss(fake_boundary, real_labels)
        
        # 内容质量损失
        content_loss = self.bce_loss(fake_content, real_labels)
        
        # 总损失
        total_loss = (self.adversarial_weight * adversarial_loss +
                     self.boundary_weight * boundary_loss +
                     self.content_weight * content_loss)
        
        return total_loss


# 测试代码
if __name__ == '__main__':
    # 测试判别器
    discriminator = DocumentQualityDiscriminator()
    
    # 测试数据
    real_images = torch.randn(2, 3, 256, 256)
    fake_images = torch.randn(2, 3, 256, 256)
    
    # 前向传播
    real_validity, real_boundary, real_content = discriminator(real_images)
    fake_validity, fake_boundary, fake_content = discriminator(fake_images)
    
    print("判别器测试:")
    print(f"真实图像 - 有效性: {real_validity.shape}, 边界: {real_boundary.shape}, 内容: {real_content.shape}")
    print(f"生成图像 - 有效性: {fake_validity.shape}, 边界: {fake_boundary.shape}, 内容: {fake_content.shape}")
    
    # 测试损失函数
    d_loss_fn = DiscriminatorLoss()
    g_loss_fn = GeneratorAdversarialLoss()
    
    d_loss, d_losses = d_loss_fn(real_validity, real_boundary, real_content,
                                 fake_validity, fake_boundary, fake_content)
    g_loss = g_loss_fn(fake_validity, fake_boundary, fake_content)
    
    print(f"\n损失测试:")
    print(f"判别器损失: {d_loss.item():.4f}")
    print(f"生成器对抗损失: {g_loss.item():.4f}")
    print("✅ 判别器模块测试通过！")