# 模型架构文档

本文档详细说明了ShaDocFormer项目中的三个核心模型文件的实现流程和功能。

## 1. 整体架构概览

整个系统采用**双流架构**，包括：
- **掩码生成网络** (`mask1.py` - RestormerMask)
- **精细修复网络** (`RUNetCA.py` - RefineUNetCoord)
- **主控制器** (`model.py` - Model)

流程：输入灰度图 → 掩码网络生成二通道掩码 → 拼接原图 → U-Net精细修复

## 2. 模型详细分析

### 2.1 mask1.py - RestormerMask 模块

#### 核心功能
生成文档前景/背景分割掩码，替代传统Otsu阈值分割，采用学习式方法。
#### 输入输出
- **输入**: `bin_x` [B, 1, H, W] - 灰度图像
- **输出**: [B, 2, H, W] - 双通道掩码（学习掩码 + 修正掩码）

#### 网络结构

1. **掩码预测器 (MaskPredictor)**
```
Conv3x3(1→16, padding=1)→ReLU→Conv3x3(16→16, padding=1)→ReLU→Conv3x3(16→1, padding=1)→Sigmoid
```
- 生成初始概率掩码，值域[0,1]
- 表示每个像素属于前景的概率

2. **Transformer Backbone**
- 包含4个 `TransformerBlockRestormer` 块
- 每个块包含：
  - `LayerNorm` (带/不带偏置)
  - `ConvAttention` (卷积投影注意力)
  - `GDFN` (门控深度卷积前馈网络)
  - `EdgeAwareLayer` (边缘感知层)

#### ConvAttention (CvT-style Convolutional Projection Attention) 机制

```
输入: [B, C, H, W]
步骤:
1. 生成QKV: Conv3x3(无偏置) → [B, 3C, H, W]
2. 多头拆分: [B, num_heads, C_per_head, H*W]
3. Q,K归一化: F.normalize(q, dim=-1), F.normalize(k, dim=-1)
4. 注意力计算: softmax(QK^T * temperature) * V
5. 重塑输出: [B, C, H, W] + Conv1x1输出投影
```

#### GDFN (Gated-Dconv Feed-Forward Network)

```
输入: [B, C, H, W]
步骤:
1. 扩展通道: Conv1x1 → [B, 2*hidden, H, W] (hidden = dim * 2.66)
2. 深度卷积: DepthwiseConv3x3 (groups=2*hidden)
3. 通道分割: chunk(2, dim=1) → x1, x2 [B, hidden, H, W]
4. 门控机制: GeLU(x1) * x2
5. 压缩通道: Conv1x1 → [B, C, H, W]
```

### 2.2 RUNetCA.py - RefineUNetCoord 模块

#### 核心功能
基于U-Net架构进行图像修复和增强，集成坐标注意力机制。
#### 输入输出
- **输入**: 拼接特征图 [B, 5, H, W] (3通道原图 + 2通道掩码)
- **输出**: 修复图像 [B, 3, H, W]

#### 网络结构

1. **编码器路径**
```
输入: [B, 5, H, W]
enc1: DoubleConv(5→64) → [B, 64, H, W]
pool1: MaxPool2d(2) → [B, 64, H/2, W/2]
enc2: DoubleConv(64→128) → [B, 128, H/2, W/2]
pool2: MaxPool2d(2) → [B, 128, H/4, W/4]
```

2. **Bottleneck**
```
bottleneck: DoubleConv(128→256) → [B, 256, H/4, W/4]
```

3. **解码器路径**
```
up2: Up(256+128→128) → [B, 128, H/2, W/2]
ca2: CoordAttention(128) → [B, 128, H/2, W/2]
up1: Up(128+64→64) → [B, 64, H, W]
ca1: CoordAttention(64) → [B, 64, H, W]
```

4. **输出层**
```
out_conv: Conv2d(64→3, kernel_size=1) → [B, 3, H, W]
```

3. **坐标注意力 (CoordAttention)**
- 水平方向平均池化: [B, C, H, 1]
- 垂直方向平均池化: [B, C, 1, W]
- 拼接特征: [B, C, H+W, 1]
- 卷积分解: Conv1x1 → 分为两个方向注意力
- 空间加权: 输出 = 输入 * 水平注意力 * 垂直注意力

4. **DoubleConv 模块**
```
Conv3x3→BN→ReLU → Conv3x3→BN→ReLU
```

5. **Up 模块**
```python
def forward(self, x1, x2):  # x1: 来自下层, x2: 跳跃连接
    x1 = self.up(x1)        # 上采样: Upsample(bilinear) 或 ConvTranspose2d
    # 尺寸对齐: 零填充处理尺寸差异
    x = torch.cat([x2, x1], dim=1)  # 跳跃连接: 拼接编码器对应层特征
    return self.conv(x)     # DoubleConv处理
```

**Up模块参数**:
- `bilinear=True`: 使用双线性插值上采样
- `bilinear=False`: 使用转置卷积上采样

### 2.3 model.py - 主控制器

#### 核心功能
整合掩码生成网络和修复网络，完成端到端的文档图像增强。

#### Model 类结构
```python
class Model(nn.Module):
    def __init__(self):
        self.mask = RestormerMask()           # 掩码生成网络
        self.refine = RefineUNetCoord(bilinear=True)  # 精细修复网络
```

#### forward 方法签名
```python
def forward(self, bin_x, x):
    """
    Args:
        bin_x : [B, 1, H, W]   # 输入的灰度图像（阈值分割用）
        x     : [B, 3, H, W]   # 输入的原始RGB图像
    Returns:
        res   : [B, 3, H, W]   # 输出的修复/增强图像
    """
```

#### 完整处理流程

```
输入:
  bin_x: [B, 1, H, W] - 灰度图像（用于掩码生成）
  x: [B, 3, H, W] - 原始RGB图像

处理流程:
1. 掩码生成: mask = self.mask(bin_x)  # [B, 1, H, W] → [B, 2, H, W]
   # RestormerMask输出双通道掩码（初始掩码 + Transformer修正掩码）

2. 特征拼接: x_res = torch.cat((mask, x), dim=1)  # [B, 5, H, W]
   # mask: [B, 2, H, W] + x: [B, 3, H, W] → x_res: [B, 5, H, W]

3. 精细修复: res = self.refine(x_res)  # [B, 5, H, W] → [B, 3, H, W]
   # RefineUNetCoord基于5通道输入进行图像修复

输出: res [B, 3, H, W] - 修复/增强后的RGB图像
```

#### 数据流向图
```
bin_x [B,1,H,W] ──→ RestormerMask ──→ mask [B,2,H,W]
                                           ↓
x [B,3,H,W] ────────────────────→ concat ──→ x_res [B,5,H,W]
                                           ↓
                              RefineUNetCoord ──→ res [B,3,H,W]
```

## 3. 技术亮点

### 3.1 双流架构优势
- **掩码网络**: 学习式分割替代固定阈值
- **修复网络**: 基于掩码进行条件增强
- **端到端训练**: 联合优化两个子网络

### 3.2 关键创新点

1. **ConvAttention**: CvT风格的卷积投影注意力，结合卷积和自注意力优势
2. **GDFN前馈**: 门控深度卷积前馈网络，高效的特征变换
3. **双掩码设计**: 初始掩码 + Transformer修正掩码，渐进式优化
4. **边缘感知**: 集成EdgeAwareLayer，保持文档结构完整性
5. **可学习温度**: 注意力计算中的可学习温度参数，自适应调节注意力强度

#### TransformerBlockRestormer 执行顺序
```
输入: [B, C, H, W]
1. 注意力分支: x = x + ConvAttention(LayerNorm(x))
2. 边缘感知: x = x + EdgeAwareLayer(x)  
3. 前馈分支: x = x + GDFN(LayerNorm(x))
输出: [B, C, H, W]
```

### 3.3 损失函数策略（推测）
- L1/L2重建损失
- 感知损失（VGG特征）
- 掩码预测二分类损失
- 对抗损失（如使用GAN）

## 4. 训练策略

### 4.1 数据预处理
- 灰度图转换: RGB → Grayscale (用于bin_x输入)
- 尺寸归一化: 512×512 或自适应
- 数据增强: 翻转、旋转、亮度调整

### 4.4 模型测试示例
```python
# 基于model.py中的测试代码
bin_x = torch.randn(1, 1, 512, 512).cuda()  # 灰度图输入
x = torch.randn(1, 3, 512, 512).cuda()      # RGB图输入

model = Model().cuda()
output = model(bin_x, x)

print(f'bin_x size: {bin_x.size()}')        # [1, 1, 512, 512]
print(f'x size: {x.size()}')                # [1, 3, 512, 512]
print(f'output size: {output.size()}')      # [1, 3, 512, 512]
```

### 4.2 训练流程
1. **阶段一**: 单独训练掩码网络
2. **阶段二**: 冻结掩码网络，训练修复网络
3. **阶段三**: 联合微调两个网络

### 4.3 硬件要求
- GPU内存: ≥8GB（推荐16GB）
- 批量大小: 根据显存调整
- 混合精度: 可选，加速训练

## 5. 应用场景

### 5.1 文档图像处理
- **去噪**: 去除扫描噪声和伪影
- **二值化**: 改进传统Otsu阈值方法
- **增强**: 提升文字清晰度和对比度
- **修复**: 填补缺失或损坏区域

### 5.2 性能优势
- **自适应**: 数据驱动的参数学习
- **鲁棒性**: 对光照变化不敏感
- **保边性**: 保持文字边缘锐利度
- **实时性**: 轻量级架构设计


### 6.1 数据集
- RDD
- Kligler
