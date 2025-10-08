import torch
import torch.nn as nn
import torch.nn.functional as F
from .document_boundary import DocumentBoundaryAttention


class DoubleConv(nn.Module):
    """两个 3×3 卷积 + BN + ReLU"""
    def __init__(self, in_ch, out_ch, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CoordAttention(nn.Module):
    """Coordinate Attention (Hou et al., 2021)"""
    def __init__(self, in_channels, reduction=32):
        super(CoordAttention, self).__init__()
        mid_channels = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.size()
        x_h = F.adaptive_avg_pool2d(x, (H, 1))  # [B, C, H, 1]
        x_w = F.adaptive_avg_pool2d(x, (1, W))  # [B, C, 1, W]
        x_w = x_w.permute(0, 1, 3, 2)           # [B, C, W, 1]

        y = torch.cat([x_h, x_w], dim=2)        # [B, C, H+W, 1]
        y = self.act(self.bn1(self.conv1(y)))

        y_h, y_w = torch.split(y, [H, W], dim=2)
        attn_h = torch.sigmoid(self.conv_h(y_h))    # [B, C, H, 1]
        attn_w = torch.sigmoid(self.conv_w(y_w))    # [B, C, W, 1]
        attn_w = attn_w.permute(0, 1, 3, 2)         # [B, C, 1, W]

        return x * attn_h * attn_w


class Up(nn.Module):
    """U-Net 解码器中的上采样块 + 跳跃连接 + DoubleConv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class RefineUNetCoord(nn.Module):
    """
    U-Net + CoordAttention + DocumentBoundaryAttention 版本
    输入: [B, 5, H, W]
    输出: [B, 3, H, W]
    """
    def __init__(self, bilinear=True, use_document_boundary=True):
        super(RefineUNetCoord, self).__init__()
        
        self.use_document_boundary = use_document_boundary

        # 编码路径
        self.enc1 = DoubleConv(5, 64)       # 5 -> 64
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)     # 64 -> 128
        self.pool2 = nn.MaxPool2d(2)

        # bottleneck
        self.bottleneck = DoubleConv(128, 256)  # 128 -> 256

        # 解码路径
        self.up2 = Up(256 + 128, 128, bilinear)  # 256+128 -> 128
        self.ca2 = CoordAttention(128)
        
        # 文档边界注意力模块 - 在第二层解码后应用
        if self.use_document_boundary:
            self.doc_boundary2 = DocumentBoundaryAttention(128)

        self.up1 = Up(128 + 64, 64, bilinear)   # 128+64 -> 64
        self.ca1 = CoordAttention(64)
        
        # 文档边界注意力模块 - 在最后一层解码后应用（主要的边界优化）
        if self.use_document_boundary:
            self.doc_boundary1 = DocumentBoundaryAttention(64)

        # 输出
        self.out_conv = nn.Conv2d(64, 3, kernel_size=1)  # 64 -> 3

    def forward(self, x):
        # 编码路径
        e1 = self.enc1(x)                     # [B, 64, H, W]
        e2 = self.enc2(self.pool1(e1))        # [B, 128, H/2, W/2]
        b = self.bottleneck(self.pool2(e2))   # [B, 256, H/4, W/4]

        # 解码路径 + 坐标注意力
        d2 = self.ca2(self.up2(b, e2))        # [B, 128, H/2, W/2]
        
        # 应用文档边界注意力 - 第二层
        if self.use_document_boundary:
            d2 = self.doc_boundary2(d2)       # 边界优化
        
        d1 = self.ca1(self.up1(d2, e1))       # [B, 64, H, W]
        
        # 应用文档边界注意力 - 最终层（主要的边界处理）
        if self.use_document_boundary:
            d1 = self.doc_boundary1(d1)       # 主要的边界优化
        
        return self.out_conv(d1)              # [B, 3, H, W]


# ====================== 测试 ======================
if __name__ == '__main__':
    x = torch.randn(1, 5, 512, 512)  # 输入: [B, 5, H, W]
    model = RefineUNetCoord(bilinear=True)
    y = model(x)
    print(f'Input: {x.shape}')
    print(f'Output: {y.shape}')
