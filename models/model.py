import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from models.mask import RestormerMask
from models.refine import RefineUNetCoord


class Model(nn.Module):
    def __init__(self, use_document_boundary=True):
        super(Model, self).__init__()

        self.mask = RestormerMask()

        self.refine = RefineUNetCoord(bilinear=True, 
                                     use_document_boundary=use_document_boundary)

    def forward(self, bin_x, x):
        """
        Args:
            bin_x : [B, 1, H, W]   # 输入的灰度图像（阈值分割用）
            x     : [B, 3, H, W]   # 输入的原始RGB图像

        Returns:
            res   : [B, 3, H, W]   # 输出的修复/增强图像
        """
        mask = self.mask(bin_x)
        # 输入 bin_x: [B, 1, H, W]
        # 输出 mask: [B, 2, H, W]   # 两个通道 掩码 + transformer 修正掩码

        x_res = torch.cat((mask, x), dim=1)
        # x:    [B, 3, H, W]
        # mask: [B, 2, H, W]
        # 拼接结果 x_res: [B, 5, H, W]  # 5通道作为Refiner输入

        res = self.refine(x_res)    # [B, 5, H, W] -> [B, 3, H, W]

        return res


if __name__ == '__main__':
    # 灰度图 (bin_x)，单通道
    bin_x = torch.randn(1, 1, 512, 512).cuda()  # [B, 1, H, W]

    # 彩色图 (x)
    x = torch.randn(1, 3, 512, 512).cuda()  # [B, 3, H, W]

    model = Model().cuda()
    output = model(bin_x, x)

    print(f'bin_x size: {bin_x.size()}')        # [1, 1, 512, 512]
    print(f'x size: {x.size()}')                # [1, 3, 512, 512]
    print(f'output size: {output.size()}')      # [1, 3, 512, 512]

    # model = Model().cuda()
    # img = Image.open('test.jpg').convert('RGB')
    # img = TF.to_tensor(img).cuda()
    # img = TF.resize(img, (512, 512)).unsqueeze(0)
    # g_img = TF.rgb_to_grayscale(img)
    # out = model(g_img, img)
    # print(out.shape)
