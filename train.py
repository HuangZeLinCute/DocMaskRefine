import warnings
import torch
import torch.optim as optim
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.functional.regression import mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from config import Config
from data import get_data
from models import *
from utils import *
from utils import losses
from utils.loss_scheduler import LossWeightScheduler, create_loss_scheduler_from_config

warnings.filterwarnings('ignore')


def seed_everything(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def manage_best_checkpoints(best_checkpoints, current_rmse, current_epoch, model_session, save_dir, max_keep=2):
    """
    管理最好的权重文件，只保留最好的两个
    Args:
        best_checkpoints: 当前最好的checkpoints列表 [(rmse, epoch, filepath), ...]
        current_rmse: 当前epoch的RMSE
        current_epoch: 当前epoch
        model_session: 模型会话名
        save_dir: 保存目录
        max_keep: 最多保留的checkpoint数量
    Returns:
        updated_best_checkpoints: 更新后的最好checkpoints列表
        should_save: 是否应该保存当前checkpoint
    """
    should_save = False
    
    # 如果还没有达到最大保留数量，直接添加
    if len(best_checkpoints) < max_keep:
        should_save = True
    else:
        # 找到当前最差的checkpoint
        worst_rmse = max(best_checkpoints, key=lambda x: x[0])[0]
        if current_rmse < worst_rmse:
            should_save = True
    
    if should_save:
        # 生成新的checkpoint文件路径
        new_checkpoint_path = os.path.join(save_dir, f"{model_session}_epoch_{current_epoch}.pth")
        
        # 添加到列表
        best_checkpoints.append((current_rmse, current_epoch, new_checkpoint_path))
        
        # 按RMSE排序（从小到大）
        best_checkpoints.sort(key=lambda x: x[0])
        
        # 如果超过最大保留数量，删除最差的
        while len(best_checkpoints) > max_keep:
            worst_checkpoint = best_checkpoints.pop()  # 移除最后一个（最差的）
            worst_filepath = worst_checkpoint[2]
            
            # 删除文件
            if os.path.exists(worst_filepath):
                try:
                    os.remove(worst_filepath)
                    print(f"🗑️  删除较差的权重文件: {os.path.basename(worst_filepath)} (RMSE: {worst_checkpoint[0]:.4f})")
                except Exception as e:
                    print(f"⚠️  删除文件失败 {worst_filepath}: {e}")
        
        # 打印当前保留的最好checkpoints
        print(f"📁 当前保留的最好权重文件:")
        for i, (rmse, epoch, filepath) in enumerate(best_checkpoints):
            print(f"   {i+1}. {os.path.basename(filepath)} - RMSE: {rmse:.4f} (Epoch {epoch})")
    
    return best_checkpoints, should_save


def plot_metrics(epochs, psnr_list, ssim_list, rmse_list, save_dir="."):
    """
    绘制训练指标：
    - PSNR + RMSE 在一张图
    - SSIM 单独一张图
    """
    os.makedirs(save_dir, exist_ok=True)

    # ---------------- 图1: PSNR + RMSE ----------------
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 左轴: PSNR
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("PSNR ↑", color='tab:blue')
    ax1.plot(epochs, psnr_list, 'o-', color='tab:blue', label='PSNR ↑')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 右轴: RMSE
    ax2 = ax1.twinx()
    ax2.set_ylabel("RMSE ↓", color='tab:red')
    ax2.plot(epochs, rmse_list, '^-', color='tab:red', label='RMSE ↓')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # 合并图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='best')

    plt.title("Training Metrics (PSNR & RMSE)")
    fig.tight_layout()
    save_path1 = os.path.join(save_dir, "metrics_psnr_rmse.png")
    plt.savefig(save_path1)
    plt.close()
    print(f"PSNR+RMSE 图已保存到 {save_path1}")

    # ---------------- 图2: SSIM ----------------
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, ssim_list, 's-', color='tab:orange', label='SSIM ↑')
    plt.xlabel("Epoch")
    plt.ylabel("SSIM ↑")
    plt.title("Training Metrics (SSIM)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best')
    save_path2 = os.path.join(save_dir, "metrics_ssim.png")
    plt.savefig(save_path2)
    plt.close()
    print(f"SSIM 图已保存到 {save_path2}")




def train():
    # 配置
    opt = Config('config.yml')
    seed_everything(opt.OPTIM.SEED)

    # 初始化 Accelerator
    if getattr(opt.OPTIM, "WANDB", False):
        accelerator = Accelerator(log_with="wandb")
        accelerator.init_trackers(project_name=getattr(opt.OPTIM, "WANDB_PROJECT", "default_project"))
    else:
        accelerator = Accelerator()

    if accelerator.is_local_main_process:
        os.makedirs(opt.TRAINING.SAVE_DIR, exist_ok=True)

    # 数据加载
    train_dataset = get_data(opt.TRAINING.TRAIN_DIR, opt.MODEL.INPUT, opt.MODEL.TARGET, 'train', opt.TRAINING.ORI,
                             {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H})
    trainloader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True,
                             num_workers=opt.TRAINING.NUM_WORKERS, drop_last=False, pin_memory=True)

    val_dataset = get_data(opt.TRAINING.VAL_DIR, opt.MODEL.INPUT, opt.MODEL.TARGET, 'test', opt.TRAINING.ORI,
                           {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H})
    testloader = DataLoader(dataset=val_dataset, batch_size=opt.TRAINING.VAL_BATCH_SIZE, shuffle=False,
                            num_workers=opt.TRAINING.VAL_NUM_WORKERS, drop_last=False, pin_memory=True)

    # 模型与损失
    model = Model()
    
    # 从config获取损失配置
    loss_config = opt.get_loss_config()
    
    print(f"使用损失配置: {opt.LOSS.TYPE}")
    print(f"配置参数: {loss_config}")
    
    # 使用改进的损失函数，针对阴影边缘黑边问题优化
    criterion = losses.ReconstructionLoss(**loss_config)
    
    # 创建损失权重调度器
    loss_scheduler = None
    try:
        if getattr(opt.LOSS_SCHEDULER, 'ENABLE', False):
            loss_scheduler = create_loss_scheduler_from_config(opt, opt.OPTIM.NUM_EPOCHS)
            print("✅ 损失权重调度器已启用")
        else:
            print("ℹ️  损失权重调度器未启用")
    except Exception as e:
        print(f"⚠️  损失权重调度器初始化失败，将使用固定权重: {e}")
        loss_scheduler = None

    # 优化器与调度器
    optimizer_b = optim.AdamW(model.parameters(), lr=opt.OPTIM.LR_INITIAL, betas=(0.9, 0.999),
                              eps=1e-8, weight_decay=0.01)
    scheduler_b = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_b, opt.OPTIM.NUM_EPOCHS, eta_min=opt.OPTIM.LR_MIN
    )

    start_epoch = 1
    best_epoch = 1
    best_rmse = 100

    # 早停参数
    patience = getattr(opt.TRAINING, "PATIENCE", 20)
    patience_counter = 0
    
    # 保留最好的两个权重文件
    best_checkpoints = []  # 存储最好的两个checkpoint信息 [(rmse, epoch, filepath), ...]

    # checkpoint
    resume_path = getattr(opt.TRAINING, "RESUME", None)
    if resume_path and os.path.exists(resume_path):
        print(f"=> 加载checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location="cpu")

        state_dict = checkpoint['state_dict']
        # 如果有 "module." 前缀，则去掉
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[len("module."):]] = v
            else:
                new_state_dict[k] = v

        # 加载修改后的 state_dict
        model.load_state_dict(new_state_dict)

        
        if 'optimizer_state_dict' in checkpoint:
            optimizer_b.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_rmse = checkpoint.get('best_rmse', best_rmse)
            best_epoch = checkpoint.get('best_epoch', best_epoch)
            
            # 恢复损失调度器状态
            if loss_scheduler is not None:
                if 'loss_scheduler_state' in checkpoint:
                    # 从checkpoint恢复调度器状态
                    try:
                        loss_scheduler.current_weights = checkpoint['loss_scheduler_state']['current_weights']
                        loss_scheduler.best_metric = checkpoint['loss_scheduler_state']['best_metric']
                        loss_scheduler.patience_counter = checkpoint['loss_scheduler_state']['patience_counter']
                        loss_scheduler.metric_history = checkpoint['loss_scheduler_state']['metric_history']
                        print(f"✅ 损失调度器状态已从checkpoint恢复")
                    except Exception as e:
                        print(f"⚠️  损失调度器状态恢复失败: {e}")
                else:
                    # 旧checkpoint没有调度器状态，根据当前epoch推断应有的权重
                    print(f"ℹ️  旧checkpoint未包含调度器状态，根据epoch {start_epoch-1} 推断权重...")
                    inferred_weights = loss_scheduler.step(start_epoch - 1, best_rmse)
                    loss_scheduler.current_weights = inferred_weights
                    criterion.update_weights(inferred_weights)
                    print(f"✅ 调度器权重已推断并应用")
                    
                    # 打印推断的权重
                    weights_str = ", ".join([f"{k}:{v:.3f}" for k, v in inferred_weights.items()])
                    print(f"   📊 推断的损失权重: {weights_str}")
            
            print(f"=> 从第 {start_epoch} 轮继续训练 (best_rmse={best_rmse:.4f}, best_epoch={best_epoch})")
        else:
            print("=> 只找到模型参数，将以fine-tune模式从头训练优化器")
    else:
        print("=> 未找到checkpoint，将从头训练")

    # prepare
    trainloader, testloader = accelerator.prepare(trainloader, testloader)
    model = accelerator.prepare(model)
    optimizer_b, scheduler_b = accelerator.prepare(optimizer_b, scheduler_b)

    size = len(testloader)

    # 记录指标
    epoch_list, psnr_list, ssim_list, rmse_list = [], [], [], []
    
    # 创建或打开RMSE记录文件
    rmse_log_path = os.path.join(opt.TRAINING.SAVE_DIR, f"{opt.MODEL.SESSION}_rmse_log.txt")
    if accelerator.is_local_main_process:
        # 如果是断点续训，追加模式；否则新建文件
        log_mode = 'a' if (resume_path and os.path.exists(resume_path)) else 'w'
        with open(rmse_log_path, log_mode) as f:
            if log_mode == 'w':
                f.write("Epoch,PSNR,SSIM,RMSE\n")  # 写入表头
            else:
                f.write(f"\n# Resumed training from epoch {start_epoch}\n")
        print(f"📝 RMSE日志将保存到: {rmse_log_path}")

    try:
        for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
            # 更新损失权重（在每个epoch开始时）
            if loss_scheduler is not None:
                # 如果有验证指标历史，使用最新的RMSE
                validation_metric = best_rmse if epoch > start_epoch else None
                updated_weights = loss_scheduler.step(epoch, validation_metric)
                criterion.update_weights(updated_weights)
            
            model.train()
            for _, data in enumerate(tqdm(trainloader, disable=not accelerator.is_local_main_process)):
                inp, gray, tar = data[0].contiguous(), data[1].contiguous(), data[2]
                optimizer_b.zero_grad()

                # forward
                res = model(gray, inp)

                # 使用简化的损失函数（只包含MSE和SSIM）
                train_loss, loss_mse, loss_ssim = criterion(res, tar)

                # backward
                accelerator.backward(train_loss)
                optimizer_b.step()
            scheduler_b.step()

            # 验证
            if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
                model.eval()
                psnr, ssim, rmse = 0, 0, 0
                for _, data in enumerate(tqdm(testloader, disable=not accelerator.is_local_main_process)):
                    inp, gray, tar = data[0].contiguous(), data[1].contiguous(), data[2]
                    with torch.no_grad():
                        res = model(gray, inp)
                    res, tar = accelerator.gather((res, tar))
                    psnr += peak_signal_noise_ratio(res, tar, data_range=1).item()
                    ssim += structural_similarity_index_measure(res, tar, data_range=1).item()
                    rmse += mean_squared_error(torch.mul(res, 255).flatten(),
                                               torch.mul(tar, 255).flatten(),
                                               squared=False).item()
                psnr /= size
                ssim /= size
                rmse /= size

                # 记录
                epoch_list.append(epoch)
                psnr_list.append(psnr)
                ssim_list.append(ssim)
                rmse_list.append(rmse)

                # 管理最好的权重文件（只保留最好的两个）
                best_checkpoints, should_save = manage_best_checkpoints(
                    best_checkpoints, rmse, epoch, opt.MODEL.SESSION, opt.TRAINING.SAVE_DIR, max_keep=2
                )
                
                # 保存模型 & 早停
                if should_save:
                    checkpoint_data = {
                        'state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer_b.state_dict(),
                        'epoch': epoch,
                        'best_rmse': min(rmse, best_rmse),
                        'best_epoch': epoch if rmse < best_rmse else best_epoch,
                    }
                    
                    # 保存损失调度器状态
                    if loss_scheduler is not None:
                        checkpoint_data['loss_scheduler_state'] = {
                            'current_weights': loss_scheduler.current_weights,
                            'best_metric': loss_scheduler.best_metric,
                            'patience_counter': loss_scheduler.patience_counter,
                            'metric_history': loss_scheduler.metric_history
                        }
                    
                    save_checkpoint(checkpoint_data, epoch, opt.MODEL.SESSION, opt.TRAINING.SAVE_DIR)
                
                # 更新最佳记录
                if rmse < best_rmse:
                    best_epoch = epoch
                    best_rmse = rmse
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"早停触发！连续 {patience} 个验证周期无提升 (best RMSE={best_rmse:.4f}, best epoch={best_epoch})")
                        raise KeyboardInterrupt

                # 日志（仅在启用 wandb 时才执行）
                if getattr(opt.OPTIM, "WANDB", False):
                    accelerator.log({"PSNR": psnr, "SSIM": ssim, "RMSE": rmse}, step=epoch)

                # 控制台输出
                if accelerator.is_local_main_process:
                    output_str = (f"epoch: {epoch}, RMSE:{rmse:.4f}, PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, "
                                f"best RMSE: {best_rmse:.4f}, best epoch: {best_epoch}")
                    
                    # 如果启用了损失调度器，显示当前权重
                    if loss_scheduler is not None and epoch % 10 == 0:
                        current_weights = criterion.get_weights()
                        weights_str = ", ".join([f"{k}:{v:.3f}" for k, v in current_weights.items()])
                        output_str += f"\n   📊 当前损失权重: {weights_str}"
                    
                    print(output_str)
                    
                    # 记录RMSE到文件
                    with open(rmse_log_path, 'a') as f:
                        f.write(f"{epoch},{psnr:.6f},{ssim:.6f},{rmse:.6f}\n")

    except KeyboardInterrupt:
        print("\n==> 训练停止，开始保存指标图 ...")

    # 显示最终保留的权重文件
    if accelerator.is_local_main_process and len(best_checkpoints) > 0:
        print(f"\n🏆 训练完成！最终保留的最好权重文件:")
        for i, (rmse, epoch, filepath) in enumerate(best_checkpoints):
            status = "🥇 最佳" if i == 0 else "🥈 次佳"
            print(f"   {status}: {os.path.basename(filepath)} - RMSE: {rmse:.4f} (Epoch {epoch})")
        
        # 推荐使用最佳权重进行测试
        if len(best_checkpoints) > 0:
            best_checkpoint_path = best_checkpoints[0][2]
            print(f"\n💡 推荐使用最佳权重进行测试:")
            print(f"   python test.py TESTING.WEIGHT \"{best_checkpoint_path}\"")

    # 绘制指标图
    if len(epoch_list) > 0:
        plot_metrics(epoch_list, psnr_list, ssim_list, rmse_list,
             save_dir=opt.TRAINING.SAVE_DIR)

    accelerator.end_training()


if __name__ == '__main__':
    train()
