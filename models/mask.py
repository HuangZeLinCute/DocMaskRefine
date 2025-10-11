import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


# ========================
# LayerNorm (BiasFree / WithBias)
# ========================
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        # [N, C, H, W]
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        # [N, C, H, W] -> [N, H*W, C]
        n, c, h, w = x.shape
        input_x = x.permute(0, 2, 3, 1).reshape(n, -1, c)
        out = self.body(input_x)
        # [N, H*W, C] -> [N, C, H, W]
        out = out.view(n, h, w, c).permute(0, 3, 1, 2)
        return out


# ========================
# ConvAttention (CvT-style Convolutional Projection Attention)
# ========================
class ConvAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(ConvAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # CvT 卷积投影生成 QKV
        self.qkv = nn.Conv2d(
            dim, dim * 3,
            kernel_size=3, stride=1, padding=1,
            groups=1, bias=False
        )

        # 输出投影
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(self, x):
        """
        输入:  [N, C, H, W]
        输出:  [N, C, H, W]
        """
        n, c, h, w = x.shape

        # [N, C, H, W] -> [N, 3C, H, W]
        qkv = self.qkv(x)
        # -> [N, C, H, W], [N, C, H, W], [N, C, H, W]
        q, k, v = qkv.chunk(3, dim=1)

        # 多头展开
        # [N, C, H, W] -> [N, num_heads, C_head, HW]
        q = einops.rearrange(q, 'n (head c) h w -> n head c (h w)', head=self.num_heads)
        k = einops.rearrange(k, 'n (head c) h w -> n head c (h w)', head=self.num_heads)
        v = einops.rearrange(v, 'n (head c) h w -> n head c (h w)', head=self.num_heads)

        # 归一化
        q = F.normalize(q, dim=-1)   # [N, head, C_head, HW]
        k = F.normalize(k, dim=-1)   # [N, head, C_head, HW]

        # 注意力
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        # [N, head, HW, HW]
        attn = attn.softmax(dim=-1)

        # 聚合
        out = torch.matmul(attn, v)  # [N, head, C_head, HW]

        # [N, head, C_head, HW] -> [N, C, H, W]
        out = einops.rearrange(
            out, 'n head c (h w) -> n (head c) h w',
            head=self.num_heads, h=h, w=w
        )

        # 输出投影 [N, C, H, W]
        out = self.project_out(out)
        return out


# ========================
# GDFN (Gated-Dconv Feed-Forward Network)
# ========================
class GDFN(nn.Module):
    def __init__(self, dim, expansion_factor=2.66, bias=True):
        super(GDFN, self).__init__()
        hidden_features = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2,
                                kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # [N, C, H, W] -> [N, 2*hidden, H, W]
        x = self.project_in(x)
        # -> [N, hidden, H, W], [N, hidden, H, W]
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        # [N, hidden, H, W]
        x = F.gelu(x1) * x2
        # -> [N, C, H, W]
        x = self.project_out(x)
        return x


# ========================
# Transformer Block
# ========================
class TransformerBlockRestormer(nn.Module):
    def __init__(self, dim, num_heads, expansion=2.66, LayerNorm_type='WithBias'):
        super(TransformerBlockRestormer, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = ConvAttention(dim, num_heads)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = GDFN(dim, expansion)

    def forward(self, x):
        # [N, C, H, W] -> 残差注意力
        x = x + self.attn(self.norm1(x))

        # -> 残差 FFN
        x = x + self.ffn(self.norm2(x))
        return x


# ========================
# MaskPredictor
# ========================
class MaskPredictor(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_dim=16):
        super(MaskPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_dim, base_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_dim, out_ch, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)  # [N, 1, H, W]


# ========================
# AdaptiveShadowMaskGenerator
# ========================
class AdaptiveShadowMaskGenerator(nn.Module):
    def __init__(self, base_dim=32, num_blocks=4, num_heads=8, expansion=2.66):
        super(AdaptiveShadowMaskGenerator, self).__init__()
        self.mask_predictor = MaskPredictor(in_ch=1, out_ch=1, base_dim=16)

        self.conv_in = nn.Conv2d(1, base_dim, kernel_size=3, padding=1, bias=True)
        self.blocks = nn.ModuleList([
            TransformerBlockRestormer(base_dim, num_heads=num_heads, expansion=expansion)
            for _ in range(num_blocks)
        ])
        self.conv_out = nn.Conv2d(base_dim, 1, kernel_size=3, padding=1, bias=True)

    def forward(self, bin_x):
        mask = self.mask_predictor(bin_x)  # [N, 1, H, W]

        x = self.conv_in(mask)             # [N, base_dim, H, W]
        for blk in self.blocks:
            x = blk(x)                     # [N, base_dim, H, W]
        trans_mask = torch.sigmoid(self.conv_out(x))  # [N, 1, H, W]

        res_mask = torch.cat((mask, trans_mask), dim=1)  # [N, 2, H, W]
        return res_mask


if __name__ == '__main__':
    input_tensor = torch.randn(1, 1, 512, 512).cuda()
    model = AdaptiveShadowMaskGenerator().cuda()
    output = model(input_tensor).cuda()
    print(f'Input size: {input_tensor.size()}')
    print(f'Output size: {output.size()}')
