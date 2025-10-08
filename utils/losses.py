import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure
import torchvision.models as models


class EdgeAwareLoss(nn.Module):
    """边缘感知损失 - 专门处理阴影边界黑边问题"""
    
    def __init__(self):
        super(EdgeAwareLoss, self).__init__()
        # Sobel算子用于边缘检测
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def get_edge_map(self, img):
        """提取边缘图"""
        # 转为灰度图
        gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        
        # 确保Sobel卷积核在正确的设备上
        sobel_x = self.sobel_x.to(img.device)
        sobel_y = self.sobel_y.to(img.device)
        
        # 计算梯度
        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        
        # 梯度幅值
        edge_map = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        return edge_map
    
    def forward(self, pred, target):
        """
        计算边缘感知损失
        重点关注边缘区域的重建质量
        """
        # 获取边缘图
        edge_pred = self.get_edge_map(pred)
        edge_target = self.get_edge_map(target)
        
        # 边缘损失
        edge_loss = F.mse_loss(edge_pred, edge_target)
        
        return edge_loss


class GradientLoss(nn.Module):
    """梯度损失 - 保持图像平滑过渡，减少黑边"""
    
    def __init__(self):
        super(GradientLoss, self).__init__()
    
    def forward(self, pred, target):
        """
        计算梯度损失，确保预测图像的梯度与目标图像一致
        """
        # 计算水平和垂直梯度
        pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        
        target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        # 梯度损失
        loss_grad_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_grad_y = F.l1_loss(pred_grad_y, target_grad_y)
        
        return loss_grad_x + loss_grad_y


class TransparencyLoss(nn.Module):
    """透明度损失 - 让阴影边界产生透明过渡效果，解决黑边问题"""
    
    def __init__(self, fade_width=10, min_alpha=0.3):
        super(TransparencyLoss, self).__init__()
        self.fade_width = fade_width
        self.min_alpha = min_alpha
        
        # 用于边缘检测的Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def detect_edges(self, img):
        """检测图像边缘"""
        # 转为灰度图
        gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        
        # 确保Sobel卷积核在正确的设备上
        sobel_x = self.sobel_x.to(img.device)
        sobel_y = self.sobel_y.to(img.device)
        
        # 计算梯度
        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        
        # 梯度幅值
        edge_map = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        return edge_map
    
    def create_gradient_mask(self, edge_map):
        """
        从边缘图创建渐变透明掩码
        Args:
            edge_map: [B, 1, H, W] 边缘图
        Returns:
            gradient_mask: [B, 1, H, W] 渐变掩码 (min_alpha到1.0)
        """
        # 二值化边缘图
        edge_threshold = 0.1
        binary_edges = (edge_map > edge_threshold).float()
        
        # 使用最大池化扩展边缘区域
        kernel_size = self.fade_width
        if kernel_size % 2 == 0:
            kernel_size += 1  # 确保是奇数
            
        # 创建扩展的边缘区域
        expanded_edges = F.max_pool2d(binary_edges, kernel_size, stride=1, padding=kernel_size//2)
        
        # 创建距离场 - 使用高斯模糊来创建平滑的渐变
        # 距离越远，透明度越高（alpha值越大）
        blur_kernel_size = max(3, self.fade_width // 2)
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1
            
        # 创建高斯核
        sigma = self.fade_width / 3.0
        x = torch.arange(blur_kernel_size, dtype=torch.float32, device=edge_map.device)
        x = x - blur_kernel_size // 2
        gauss_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        
        # 2D高斯核
        gauss_2d = gauss_1d.view(1, -1) * gauss_1d.view(-1, 1)
        gauss_2d = gauss_2d.view(1, 1, blur_kernel_size, blur_kernel_size)
        
        # 应用高斯模糊创建渐变
        blurred_edges = F.conv2d(expanded_edges, gauss_2d, padding=blur_kernel_size//2)
        
        # 将渐变映射到 [min_alpha, 1.0] 范围
        # 边缘处 = min_alpha, 远离边缘 = 1.0
        gradient_mask = self.min_alpha + (1.0 - self.min_alpha) * (1.0 - blurred_edges)
        
        return gradient_mask
    
    def forward(self, pred, target):
        """
        计算透明度损失
        Args:
            pred: [B, 3, H, W] 预测图像
            target: [B, 3, H, W] 目标图像
        Returns:
            transparency_loss: 透明度损失值
        """
        # 检测预测图像和目标图像的边缘
        pred_edges = self.detect_edges(pred)
        target_edges = self.detect_edges(target)
        
        # 使用目标图像的边缘创建渐变掩码（更准确）
        gradient_mask = self.create_gradient_mask(target_edges)
        
        # 应用透明度效果：在边缘区域混合预测和目标
        # transparent_pred = pred * gradient_mask + target * (1 - gradient_mask)
        # 但我们希望预测图像在边缘处更接近目标，所以直接约束预测值
        
        # 在边缘区域，预测值应该更接近目标值
        edge_weight = 1.0 - gradient_mask  # 边缘处权重更高
        weighted_diff = edge_weight * (pred - target) ** 2
        
        transparency_loss = weighted_diff.mean()
        
        return transparency_loss


class PerceptualLoss(nn.Module):
    """
    感知损失 - 使用预训练VGG网络提取特征
    防止颜色偏移，保持视觉自然度
    """
    
    def __init__(self, feature_layers=[2, 7, 12, 21], use_gpu=True):
        super(PerceptualLoss, self).__init__()
        
        try:
            # 使用预训练的VGG16
            vgg = models.vgg16(pretrained=True)
            self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:max(feature_layers)+1])
            
            # 冻结VGG参数
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
                
            self.feature_layers = feature_layers
            self.criterion = nn.MSELoss()
            
            # VGG预处理的均值和标准差
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            
            self.vgg_available = True
            
        except Exception as e:
            print(f"警告: VGG模型加载失败，使用简化的感知损失: {e}")
            self.vgg_available = False
            self.criterion = nn.MSELoss()
        
    def preprocess(self, x):
        """VGG预处理"""
        # 确保输入在[0,1]范围内
        x = torch.clamp(x, 0, 1)
        # 确保均值和标准差在正确的设备上
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        # 标准化
        x = (x - mean) / std
        return x
        
    def forward(self, pred, target):
        """
        计算感知损失
        Args:
            pred: [B, 3, H, W] 预测图像
            target: [B, 3, H, W] 目标图像
        Returns:
            perceptual_loss: 感知损失值
        """
        if not self.vgg_available:
            # 如果VGG不可用，使用简化的感知损失（基于颜色空间的损失）
            # 转换到YUV颜色空间，更关注亮度差异
            pred_yuv = self.rgb_to_yuv(pred)
            target_yuv = self.rgb_to_yuv(target)
            return self.criterion(pred_yuv, target_yuv)
        
        # 确保特征提取器在正确的设备上
        if not hasattr(self, '_device_checked') or self._device_checked != pred.device:
            self.feature_extractor = self.feature_extractor.to(pred.device)
            self._device_checked = pred.device
        
        # 预处理
        pred = self.preprocess(pred)
        target = self.preprocess(target)
        
        # 提取特征
        pred_features = []
        target_features = []
        
        x_pred = pred
        x_target = target
        
        for i, layer in enumerate(self.feature_extractor):
            x_pred = layer(x_pred)
            x_target = layer(x_target)
            
            if i in self.feature_layers:
                pred_features.append(x_pred)
                target_features.append(x_target)
        
        # 计算各层特征损失
        perceptual_loss = 0
        for pred_feat, target_feat in zip(pred_features, target_features):
            perceptual_loss += self.criterion(pred_feat, target_feat)
            
        return perceptual_loss / len(self.feature_layers)
    
    def rgb_to_yuv(self, rgb):
        """RGB转YUV颜色空间"""
        # YUV转换矩阵
        rgb_to_yuv_kernel = torch.tensor([
            [0.299, 0.587, 0.114],
            [-0.14713, -0.28886, 0.436],
            [0.615, -0.51499, -0.10001]
        ], device=rgb.device, dtype=rgb.dtype)
        
        # 重塑RGB为 [B*H*W, 3]
        B, C, H, W = rgb.shape
        rgb_flat = rgb.permute(0, 2, 3, 1).reshape(-1, 3)
        
        # 应用转换
        yuv_flat = torch.matmul(rgb_flat, rgb_to_yuv_kernel.T)
        
        # 重塑回 [B, 3, H, W]
        yuv = yuv_flat.reshape(B, H, W, 3).permute(0, 3, 1, 2)
        
        return yuv


class HistogramLoss(nn.Module):
    """
    直方图损失 - 确保全局颜色分布一致
    特别适用于文档阴影去除，保持背景颜色一致性
    """
    
    def __init__(self, num_bins=256, normalize=True):
        super(HistogramLoss, self).__init__()
        self.num_bins = num_bins
        self.normalize = normalize
        
    def compute_histogram(self, x, num_bins=256):
        """
        计算图像直方图
        Args:
            x: [B, C, H, W] 输入图像
            num_bins: 直方图bin数量
        Returns:
            hist: [B, C, num_bins] 直方图
        """
        B, C, H, W = x.shape
        device = x.device
        
        # 将像素值映射到[0, num_bins-1]
        x_scaled = torch.clamp(x * (num_bins - 1), 0, num_bins - 1).long()
        
        # 计算每个通道的直方图
        histograms = []
        for c in range(C):
            channel_data = x_scaled[:, c].view(B, -1)  # [B, H*W]
            
            # 为每个batch计算直方图
            batch_hists = []
            for b in range(B):
                hist = torch.histc(channel_data[b].float(), bins=num_bins, min=0, max=num_bins-1)
                # 确保直方图在正确的设备上
                hist = hist.to(device)
                batch_hists.append(hist)
            
            channel_hist = torch.stack(batch_hists, dim=0)  # [B, num_bins]
            histograms.append(channel_hist)
        
        hist = torch.stack(histograms, dim=1)  # [B, C, num_bins]
        
        # 归一化
        if self.normalize:
            hist = hist / (H * W)
            
        return hist
    
    def forward(self, pred, target):
        """
        计算直方图损失
        Args:
            pred: [B, 3, H, W] 预测图像
            target: [B, 3, H, W] 目标图像
        Returns:
            histogram_loss: 直方图损失值
        """
        # 计算直方图
        pred_hist = self.compute_histogram(pred, self.num_bins)
        target_hist = self.compute_histogram(target, self.num_bins)
        
        # 使用L1损失比较直方图
        histogram_loss = F.l1_loss(pred_hist, target_hist)
        
        return histogram_loss


class ShadowBoundaryLoss(nn.Module):
    """阴影边界损失 - 针对阴影边界区域的加权损失"""
    
    def __init__(self, boundary_width=5):
        super(ShadowBoundaryLoss, self).__init__()
        self.boundary_width = boundary_width
    
    def get_shadow_boundary_mask(self, shadow_mask):
        """
        从阴影掩码中提取边界区域
        Args:
            shadow_mask: [B, 1, H, W] 阴影掩码 (0-1之间)
        Returns:
            boundary_mask: [B, 1, H, W] 边界掩码
        """
        # 二值化阴影掩码
        binary_mask = (shadow_mask > 0.5).float()
        
        # 形态学操作提取边界
        kernel_size = self.boundary_width
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=shadow_mask.device)
        
        # 膨胀操作
        dilated = F.conv2d(binary_mask, kernel, padding=kernel_size//2)
        dilated = (dilated > 0).float()
        
        # 腐蚀操作
        eroded = F.conv2d(binary_mask, kernel, padding=kernel_size//2)
        eroded = (eroded >= kernel_size * kernel_size).float()
        
        # 边界 = 膨胀 - 腐蚀
        boundary_mask = dilated - eroded
        
        return boundary_mask
    
    def forward(self, pred, target, shadow_mask):
        """
        计算阴影边界区域的加权损失
        Args:
            pred: [B, 3, H, W] 预测图像
            target: [B, 3, H, W] 目标图像  
            shadow_mask: [B, 1, H, W] 阴影掩码
        """
        # 获取边界掩码
        boundary_mask = self.get_shadow_boundary_mask(shadow_mask)
        
        # 在边界区域计算加权损失
        boundary_loss = F.mse_loss(pred * boundary_mask, target * boundary_mask)
        
        return boundary_loss


class ShadowBackgroundConsistencyLoss(nn.Module):
    """
    阴影背景一致性损失 - 确保阴影区域颜色贴近周围背景
    通过分析背景区域的颜色分布，约束阴影区域向背景颜色靠拢
    """
    
    def __init__(self, sample_radius=15, background_expand=10, color_weight=1.0, texture_weight=0.3):
        super(ShadowBackgroundConsistencyLoss, self).__init__()
        self.sample_radius = sample_radius      # 背景采样半径
        self.background_expand = background_expand  # 背景区域扩展
        self.color_weight = color_weight        # 颜色一致性权重
        self.texture_weight = texture_weight    # 纹理一致性权重
    
    def detect_shadow_regions(self, pred, target):
        """
        自动检测阴影区域（如果没有提供阴影掩码）
        通过比较预测图像和目标图像的亮度差异
        """
        # 转换为灰度图
        pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
        
        # 计算亮度差异
        brightness_diff = torch.abs(pred_gray - target_gray)
        
        # 阴影区域通常是预测图像比目标图像暗的区域
        shadow_mask = (pred_gray < target_gray - 0.1).float()
        
        # 使用形态学操作平滑掩码
        kernel_size = 5
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=pred.device) / (kernel_size * kernel_size)
        shadow_mask = F.conv2d(shadow_mask, kernel, padding=kernel_size//2)
        shadow_mask = (shadow_mask > 0.3).float()
        
        return shadow_mask
    
    def get_background_regions(self, shadow_mask):
        """
        获取背景区域掩码（非阴影区域的扩展）
        """
        # 背景区域 = 1 - 阴影区域
        background_mask = 1.0 - shadow_mask
        
        # 扩展背景区域，确保有足够的背景样本
        kernel_size = self.background_expand
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=shadow_mask.device)
        expanded_background = F.conv2d(background_mask, kernel, padding=kernel_size//2)
        expanded_background = (expanded_background > 0).float()
        
        return expanded_background
    
    def sample_background_colors(self, image, background_mask, shadow_mask):
        """
        从背景区域采样颜色统计信息
        Args:
            image: [B, 3, H, W] 输入图像
            background_mask: [B, 1, H, W] 背景区域掩码
            shadow_mask: [B, 1, H, W] 阴影区域掩码
        Returns:
            background_stats: dict 包含背景颜色统计信息
        """
        B, C, H, W = image.shape
        
        # 为每个阴影像素找到最近的背景像素颜色
        background_colors = []
        shadow_positions = []
        
        for b in range(B):
            # 获取当前batch的掩码和图像
            bg_mask = background_mask[b, 0]  # [H, W]
            sh_mask = shadow_mask[b, 0]      # [H, W]
            img = image[b]                   # [3, H, W]
            
            # 找到背景和阴影的像素位置
            bg_positions = torch.nonzero(bg_mask, as_tuple=False)  # [N_bg, 2]
            sh_positions = torch.nonzero(sh_mask, as_tuple=False)  # [N_sh, 2]
            
            if len(bg_positions) > 0 and len(sh_positions) > 0:
                # 获取背景像素的颜色
                bg_colors = img[:, bg_positions[:, 0], bg_positions[:, 1]].T  # [N_bg, 3]
                
                # 计算背景颜色的统计信息
                bg_mean = bg_colors.mean(dim=0)  # [3]
                bg_std = bg_colors.std(dim=0)    # [3]
                
                background_colors.append(bg_mean)
                shadow_positions.append(sh_positions)
        
        return background_colors, shadow_positions
    
    def forward(self, pred, target, shadow_mask=None):
        """
        计算阴影背景一致性损失
        Args:
            pred: [B, 3, H, W] 预测图像
            target: [B, 3, H, W] 目标图像
            shadow_mask: [B, 1, H, W] 阴影掩码（可选，如果不提供会自动检测）
        """
        # 如果没有提供阴影掩码，自动检测
        if shadow_mask is None:
            shadow_mask = self.detect_shadow_regions(pred, target)
        
        # 获取背景区域
        background_mask = self.get_background_regions(shadow_mask)
        
        # 计算颜色一致性损失
        color_loss = self.compute_color_consistency_loss(pred, target, background_mask, shadow_mask)
        
        # 计算纹理一致性损失
        texture_loss = self.compute_texture_consistency_loss(pred, target, background_mask, shadow_mask)
        
        # 组合损失
        total_loss = self.color_weight * color_loss + self.texture_weight * texture_loss
        
        return total_loss
    
    def compute_color_consistency_loss(self, pred, target, background_mask, shadow_mask):
        """计算颜色一致性损失"""
        B, C, H, W = pred.shape
        total_loss = 0.0
        valid_batches = 0
        
        for b in range(B):
            # 获取当前batch的数据
            pred_img = pred[b]      # [3, H, W]
            target_img = target[b]  # [3, H, W]
            bg_mask = background_mask[b, 0]  # [H, W]
            sh_mask = shadow_mask[b, 0]      # [H, W]
            
            # 计算背景区域的平均颜色
            if bg_mask.sum() > 0:
                # 目标图像的背景颜色（这是我们希望阴影区域接近的颜色）
                target_bg_colors = target_img.view(3, -1)[:, bg_mask.view(-1) > 0]  # [3, N_bg]
                target_bg_mean = target_bg_colors.mean(dim=1, keepdim=True)  # [3, 1]
                
                # 预测图像的阴影区域颜色
                if sh_mask.sum() > 0:
                    pred_shadow_colors = pred_img.view(3, -1)[:, sh_mask.view(-1) > 0]  # [3, N_sh]
                    
                    # 计算阴影区域与背景平均颜色的差异
                    color_diff = pred_shadow_colors - target_bg_mean  # [3, N_sh]
                    color_loss = (color_diff ** 2).mean()
                    
                    total_loss += color_loss
                    valid_batches += 1
        
        if valid_batches > 0:
            return total_loss / valid_batches
        else:
            return torch.tensor(0.0, device=pred.device)
    
    def compute_texture_consistency_loss(self, pred, target, background_mask, shadow_mask):
        """计算纹理一致性损失（基于局部梯度）"""
        # 计算图像梯度
        pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        
        target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        # 调整掩码尺寸以匹配梯度
        shadow_mask_x = shadow_mask[:, :, :, 1:]
        shadow_mask_y = shadow_mask[:, :, 1:, :]
        
        # 在阴影区域计算梯度差异
        if shadow_mask_x.sum() > 0:
            grad_loss_x = F.mse_loss(pred_grad_x * shadow_mask_x, target_grad_x * shadow_mask_x)
        else:
            grad_loss_x = torch.tensor(0.0, device=pred.device)
            
        if shadow_mask_y.sum() > 0:
            grad_loss_y = F.mse_loss(pred_grad_y * shadow_mask_y, target_grad_y * shadow_mask_y)
        else:
            grad_loss_y = torch.tensor(0.0, device=pred.device)
        
        return grad_loss_x + grad_loss_y


class ReconstructionLoss(nn.Module):
    """
    改进的图像重建损失函数 - 专门解决阴影边缘黑边问题
    包含：
    - MSE Loss (基础重建)
    - SSIM Loss (结构相似性)
    - Edge Loss (边缘感知)
    - Gradient Loss (梯度平滑)
    - Boundary Loss (阴影边界加权)
    - Transparency Loss (透明度过渡)
    - Perceptual Loss (感知损失)
    - Histogram Loss (直方图损失)

    支持动态权重调整
    """
    def __init__(self, 
                 mse_weight: float = 1.0,
                 ssim_weight: float = 0.2,
                 edge_weight: float = 0.3,
                 gradient_weight: float = 0.1,
                 boundary_weight: float = 0.5,
                 transparency_weight: float = 0.3,
                 perceptual_weight: float = 0.1,
                 histogram_weight: float = 0.05,
                 fade_width: int = 10,
                 min_alpha: float = 0.3):
        """
        Args:
            mse_weight: MSE损失权重
            ssim_weight: SSIM损失权重
            edge_weight: 边缘损失权重
            gradient_weight: 梯度损失权重  
            boundary_weight: 边界损失权重
            transparency_weight: 透明度损失权重
            perceptual_weight: 感知损失权重
            histogram_weight: 直方图损失权重
            fade_width: 透明渐变宽度
            min_alpha: 最小透明度值
        """
        super(ReconstructionLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.edge_loss = EdgeAwareLoss()
        self.gradient_loss = GradientLoss()
        self.boundary_loss = ShadowBoundaryLoss()
        self.transparency_loss = TransparencyLoss(fade_width=fade_width, min_alpha=min_alpha)
        self.perceptual_loss = PerceptualLoss()
        self.histogram_loss = HistogramLoss()
        
        # 初始权重
        self.weights = {
            'mse': mse_weight,
            'ssim': ssim_weight,
            'edge': edge_weight,
            'gradient': gradient_weight,
            'boundary': boundary_weight,
            'transparency': transparency_weight,
            'perceptual': perceptual_weight,
            'histogram': histogram_weight
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
        
        # 边缘感知损失
        loss_edge = self.edge_loss(pred, target)
        
        # 梯度损失
        loss_gradient = self.gradient_loss(pred, target)
        
        # 透明度损失
        loss_transparency = self.transparency_loss(pred, target)
        
        # 感知损失 (改善颜色匹配)
        loss_perceptual = self.perceptual_loss(pred, target)
        
        # 直方图损失 (颜色分布一致性)
        loss_histogram = self.histogram_loss(pred, target)
        
        # 使用动态权重计算总损失
        total_loss = (self.weights['mse'] * loss_mse + 
                     self.weights['ssim'] * loss_ssim + 
                     self.weights['edge'] * loss_edge + 
                     self.weights['gradient'] * loss_gradient +
                     self.weights['transparency'] * loss_transparency +
                     self.weights['perceptual'] * loss_perceptual +
                     self.weights['histogram'] * loss_histogram)
        
        # 如果提供了阴影掩码，添加边界损失
        if shadow_mask is not None:
            loss_boundary = self.boundary_loss(pred, target, shadow_mask)
            total_loss += self.weights['boundary'] * loss_boundary
        else:
            loss_boundary = torch.tensor(0.0, device=pred.device)

        return total_loss, loss_mse, loss_ssim, loss_edge, loss_gradient, loss_boundary, loss_transparency, loss_perceptual, loss_histogram


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