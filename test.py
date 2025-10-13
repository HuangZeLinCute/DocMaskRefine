import os
import warnings

warnings.filterwarnings('ignore')

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional.regression import mean_squared_error
from torchvision.utils import save_image
from tqdm import tqdm

from config import Config
from data import get_data
from models import *
from utils import *


def test():
    opt = Config('config.yml')
    seed_everything(opt.OPTIM.SEED)

    accelerator = Accelerator()
    device = accelerator.device

    # 先初始化 LPIPS，不做 normalize
    criterion_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type='alex', normalize=False
    ).to(device)

    # Data Loader
    val_dir = opt.TRAINING.VAL_DIR
    val_dataset = get_data(
        val_dir,
        opt.MODEL.INPUT,
        opt.MODEL.TARGET,
        'test',
        opt.TRAINING.ORI,
        {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H}
    )

    testloader = DataLoader(
        dataset=val_dataset,
        batch_size=opt.TESTING.BATCH_SIZE,
        shuffle=False,
        num_workers=opt.TESTING.NUM_WORKERS,
        drop_last=False,
        pin_memory=True
    )

    # Model & Metrics
    model = Model()
    load_checkpoint(model, opt.TESTING.WEIGHT)

    model, testloader = accelerator.prepare(model, testloader)
    model.eval()

    size = len(testloader)
    stat_psnr = 0
    stat_ssim = 0
    stat_lpips = 0
    stat_rmse = 0

    for _, test_data in enumerate(tqdm(testloader)):
        # data: [targets, inputs, filename]
        inp = test_data[0].contiguous()
        gray = test_data[1].contiguous()
        tar = test_data[2]

        with torch.no_grad():
            res = model(gray, inp)

        # 检查范围
        if res.min() < 0:
            # 已经在 [-1,1]，手动归一化到 [0,1] 再算 LPIPS
            res_lpips = (res + 1) / 2
            tar_lpips = (tar + 1) / 2
        else:
            # 已经在 [0,1]，直接用
            res_lpips = res
            tar_lpips = tar

        if opt.TESTING.SAVE_IMAGES:
            os.makedirs("result", exist_ok=True)
            save_image(res, os.path.join("result", test_data[3][0]))

        stat_psnr += peak_signal_noise_ratio(res, tar, data_range=1).item()
        stat_ssim += structural_similarity_index_measure(res, tar, data_range=1).item()
        stat_lpips += criterion_lpips(res_lpips, tar_lpips).item()
        stat_rmse += mean_squared_error(
            torch.mul(res, 255).flatten(),
            torch.mul(tar, 255).flatten(),
            squared=False
        ).item()

    stat_psnr /= size
    stat_ssim /= size
    stat_lpips /= size
    stat_rmse /= size

    print(f"RMSE: {stat_rmse}, PSNR: {stat_psnr}, SSIM: {stat_ssim}, LPIPS: {stat_lpips}")


if __name__ == '__main__':
    test()
