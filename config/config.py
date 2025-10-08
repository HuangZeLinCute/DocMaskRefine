#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 14:35:48 2019

@author: aditya
"""

r"""This module provides package-wide configuration management."""
from typing import Any, List

from yacs.config import CfgNode as CN


class Config(object):
    r"""
    A collection of all the required configuration parameters. This class is a nested dict-like
    structure, with nested keys accessible as attributes. It contains sensible default values for
    all the parameters, which may be overriden by (first) through a YAML file and (second) through
    a list of attributes and values.

    Extended Summary
    ----------------
    This class definition contains default values corresponding to ``joint_training`` phase, as it
    is the final training phase and uses almost all the configuration parameters. Modification of
    any parameter after instantiating this class is not possible, so you must override required
    parameter values in either through ``config_yaml`` file or ``config_override`` list.

    Parameters
    ----------
    config_yaml: str
        Path to a YAML file containing configuration parameters to override.
    config_override: List[Any], optional (default= [])
        A list of sequential attributes and values of parameters to override. This happens after
        overriding from YAML file.

    Examples
    --------
    Let a YAML file named "config.yaml" specify these parameters to override::

        ALPHA: 1000.0
        BETA: 0.5

    >>> _C = Config("config.yaml", ["OPTIM.BATCH_SIZE", 2048, "BETA", 0.7])
    >>> _C.ALPHA  # default: 100.0
    1000.0
    >>> _C.BATCH_SIZE  # default: 256
    2048
    >>> _C.BETA  # default: 0.1
    0.7

    Attributes
    ----------
    """

    def __init__(self, config_yaml: str, config_override: List[Any] = []):
        self._C = CN()
        self._C.GPU = [0]
        self._C.VERBOSE = False

        self._C.MODEL = CN()
        self._C.MODEL.SESSION = 'SR'
        self._C.MODEL.INPUT = 'input'
        self._C.MODEL.TARGET = 'target'
        self._C.MODEL.USE_DOCUMENT_BOUNDARY = True  # 启用文档边界注意力模块
        self._C.MODEL.USE_DISCRIMINATOR = True      # 启用GAN判别器
        self._C.MODEL.USE_BLACKEDGE_SUPPRESSOR = True  # 启用黑边抑制模块
        self._C.MODEL.USE_BACKGROUND_REFERENCE = True  # 启用背景参考引导模块

        self._C.OPTIM = CN()
        self._C.OPTIM.BATCH_SIZE = 1
        self._C.OPTIM.SEED = 3407
        self._C.OPTIM.NUM_EPOCHS = 300
        self._C.OPTIM.NEPOCH_DECAY = [100]
        self._C.OPTIM.LR_INITIAL = 0.0002
        self._C.OPTIM.LR_MIN = 0.0002
        self._C.OPTIM.BETA1 = 0.5
        self._C.OPTIM.WANDB = False
        
        # GAN训练相关参数
        self._C.OPTIM.DISCRIMINATOR_LR = 0.0001     # 判别器学习率
        self._C.OPTIM.ADVERSARIAL_WEIGHT = 0.1      # 对抗损失权重
        self._C.OPTIM.DISCRIMINATOR_START_EPOCH = 5  # 判别器开始训练的epoch

        self._C.TRAINING = CN()
        self._C.TRAINING.VAL_AFTER_EVERY = 3
        self._C.TRAINING.RESUME = ""
        self._C.TRAINING.TRAIN_DIR = 'images_dir/train'
        self._C.TRAINING.VAL_DIR = 'images_dir/val'
        self._C.TRAINING.SAVE_DIR = 'checkpoints'
        self._C.TRAINING.PS_W = 512
        self._C.TRAINING.PS_H = 512
        self._C.TRAINING.ORI = False
        self._C.TRAINING.NUM_WORKERS = 4
        self._C.TRAINING.VAL_BATCH_SIZE = 4
        self._C.TRAINING.VAL_NUM_WORKERS = 4
        self._C.TRAINING.PATIENCE = 20

        # 损失函数配置
        self._C.LOSS = CN()
        self._C.LOSS.TYPE = 'STRONG_EDGE'  # 损失配置类型: DEFAULT, STRONG_EDGE, SMOOTH_PRIORITY, LIGHTWEIGHT
        
        # 各种损失权重配置
        self._C.LOSS.CONFIGS = CN()
        
        # 默认配置
        self._C.LOSS.CONFIGS.DEFAULT = CN()
        self._C.LOSS.CONFIGS.DEFAULT.SSIM_WEIGHT = 0.2
        self._C.LOSS.CONFIGS.DEFAULT.EDGE_WEIGHT = 0.3
        self._C.LOSS.CONFIGS.DEFAULT.GRADIENT_WEIGHT = 0.1
        self._C.LOSS.CONFIGS.DEFAULT.BOUNDARY_WEIGHT = 0.5
        self._C.LOSS.CONFIGS.DEFAULT.TRANSPARENCY_WEIGHT = 0.0  # 默认不启用
        self._C.LOSS.CONFIGS.DEFAULT.PERCEPTUAL_WEIGHT = 0.05   # 感知损失权重
        self._C.LOSS.CONFIGS.DEFAULT.HISTOGRAM_WEIGHT = 0.02    # 直方图损失权重
        self._C.LOSS.CONFIGS.DEFAULT.FADE_WIDTH = 10
        self._C.LOSS.CONFIGS.DEFAULT.MIN_ALPHA = 0.3
        
        # 强化边缘配置 - 解决黑边问题
        self._C.LOSS.CONFIGS.STRONG_EDGE = CN()
        self._C.LOSS.CONFIGS.STRONG_EDGE.SSIM_WEIGHT = 0.15
        self._C.LOSS.CONFIGS.STRONG_EDGE.EDGE_WEIGHT = 0.5
        self._C.LOSS.CONFIGS.STRONG_EDGE.GRADIENT_WEIGHT = 0.2
        self._C.LOSS.CONFIGS.STRONG_EDGE.BOUNDARY_WEIGHT = 0.8
        self._C.LOSS.CONFIGS.STRONG_EDGE.TRANSPARENCY_WEIGHT = 0.0  # 暂时不启用
        self._C.LOSS.CONFIGS.STRONG_EDGE.FADE_WIDTH = 10
        self._C.LOSS.CONFIGS.STRONG_EDGE.MIN_ALPHA = 0.3
        
        # 透明边缘配置 - 专门解决黑边透明化
        self._C.LOSS.CONFIGS.TRANSPARENT_EDGE = CN()
        self._C.LOSS.CONFIGS.TRANSPARENT_EDGE.SSIM_WEIGHT = 0.15
        self._C.LOSS.CONFIGS.TRANSPARENT_EDGE.EDGE_WEIGHT = 0.4
        self._C.LOSS.CONFIGS.TRANSPARENT_EDGE.GRADIENT_WEIGHT = 0.2
        self._C.LOSS.CONFIGS.TRANSPARENT_EDGE.BOUNDARY_WEIGHT = 0.6
        self._C.LOSS.CONFIGS.TRANSPARENT_EDGE.TRANSPARENCY_WEIGHT = 0.4  # 启用透明度损失
        self._C.LOSS.CONFIGS.TRANSPARENT_EDGE.FADE_WIDTH = 15
        self._C.LOSS.CONFIGS.TRANSPARENT_EDGE.MIN_ALPHA = 0.2
        
        # 平滑优先配置
        self._C.LOSS.CONFIGS.SMOOTH_PRIORITY = CN()
        self._C.LOSS.CONFIGS.SMOOTH_PRIORITY.SSIM_WEIGHT = 0.3
        self._C.LOSS.CONFIGS.SMOOTH_PRIORITY.EDGE_WEIGHT = 0.2
        self._C.LOSS.CONFIGS.SMOOTH_PRIORITY.GRADIENT_WEIGHT = 0.3
        self._C.LOSS.CONFIGS.SMOOTH_PRIORITY.BOUNDARY_WEIGHT = 0.4
        self._C.LOSS.CONFIGS.SMOOTH_PRIORITY.TRANSPARENCY_WEIGHT = 0.2
        self._C.LOSS.CONFIGS.SMOOTH_PRIORITY.FADE_WIDTH = 12
        self._C.LOSS.CONFIGS.SMOOTH_PRIORITY.MIN_ALPHA = 0.4
        
        # 轻量配置
        self._C.LOSS.CONFIGS.LIGHTWEIGHT = CN()
        self._C.LOSS.CONFIGS.LIGHTWEIGHT.SSIM_WEIGHT = 0.2
        self._C.LOSS.CONFIGS.LIGHTWEIGHT.EDGE_WEIGHT = 0.1
        self._C.LOSS.CONFIGS.LIGHTWEIGHT.GRADIENT_WEIGHT = 0.05
        self._C.LOSS.CONFIGS.LIGHTWEIGHT.BOUNDARY_WEIGHT = 0.2
        self._C.LOSS.CONFIGS.LIGHTWEIGHT.TRANSPARENCY_WEIGHT = 0.1
        self._C.LOSS.CONFIGS.LIGHTWEIGHT.FADE_WIDTH = 8
        self._C.LOSS.CONFIGS.LIGHTWEIGHT.MIN_ALPHA = 0.5
        
        # 简化边缘配置 - 专注于核心边缘处理
        self._C.LOSS.CONFIGS.SIMPLE_EDGE = CN()
        self._C.LOSS.CONFIGS.SIMPLE_EDGE.SSIM_WEIGHT = 0.2
        self._C.LOSS.CONFIGS.SIMPLE_EDGE.EDGE_WEIGHT = 0.4
        self._C.LOSS.CONFIGS.SIMPLE_EDGE.GRADIENT_WEIGHT = 0.3  # 重点：高梯度权重
        self._C.LOSS.CONFIGS.SIMPLE_EDGE.BOUNDARY_WEIGHT = 0.0  # 不使用边界损失
        self._C.LOSS.CONFIGS.SIMPLE_EDGE.TRANSPARENCY_WEIGHT = 0.0  # 不使用透明度损失
        self._C.LOSS.CONFIGS.SIMPLE_EDGE.FADE_WIDTH = 10
        self._C.LOSS.CONFIGS.SIMPLE_EDGE.MIN_ALPHA = 0.3
        
        # 背景平衡配置 - 平衡背景一致性和边缘处理
        self._C.LOSS.CONFIGS.BACKGROUND_BALANCED = CN()
        self._C.LOSS.CONFIGS.BACKGROUND_BALANCED.SSIM_WEIGHT = 0.15  # 降低SSIM，为背景模块让路
        self._C.LOSS.CONFIGS.BACKGROUND_BALANCED.EDGE_WEIGHT = 0.5   # 提高边缘权重，对抗背景模块的平滑效应
        self._C.LOSS.CONFIGS.BACKGROUND_BALANCED.GRADIENT_WEIGHT = 0.4  # 提高梯度权重，强化边缘去除
        self._C.LOSS.CONFIGS.BACKGROUND_BALANCED.BOUNDARY_WEIGHT = 0.0
        self._C.LOSS.CONFIGS.BACKGROUND_BALANCED.TRANSPARENCY_WEIGHT = 0.0
        self._C.LOSS.CONFIGS.BACKGROUND_BALANCED.FADE_WIDTH = 10
        self._C.LOSS.CONFIGS.BACKGROUND_BALANCED.MIN_ALPHA = 0.3
        
        # 自定义配置 - 用户可以在YAML中覆盖
        self._C.LOSS.CONFIGS.CUSTOM = CN()
        self._C.LOSS.CONFIGS.CUSTOM.MSE_WEIGHT = 1.0
        self._C.LOSS.CONFIGS.CUSTOM.SSIM_WEIGHT = 0.2
        self._C.LOSS.CONFIGS.CUSTOM.EDGE_WEIGHT = 0.3
        self._C.LOSS.CONFIGS.CUSTOM.GRADIENT_WEIGHT = 0.1
        self._C.LOSS.CONFIGS.CUSTOM.BOUNDARY_WEIGHT = 0.5
        self._C.LOSS.CONFIGS.CUSTOM.TRANSPARENCY_WEIGHT = 0.0
        self._C.LOSS.CONFIGS.CUSTOM.PERCEPTUAL_WEIGHT = 0.05
        self._C.LOSS.CONFIGS.CUSTOM.HISTOGRAM_WEIGHT = 0.02

        self._C.LOSS.CONFIGS.CUSTOM.FADE_WIDTH = 10
        self._C.LOSS.CONFIGS.CUSTOM.MIN_ALPHA = 0.3

        # 损失权重调度器配置
        self._C.LOSS_SCHEDULER = CN()
        self._C.LOSS_SCHEDULER.ENABLE = False                    # 是否启用损失调度器
        self._C.LOSS_SCHEDULER.PRESET = 'shadow_removal'         # 预设类型
        self._C.LOSS_SCHEDULER.ADAPTIVE_PATIENCE = 5             # 自适应调整的耐心值
        self._C.LOSS_SCHEDULER.ADAPTIVE_FACTOR = 0.8             # 自适应调整因子
        self._C.LOSS_SCHEDULER.VERBOSE = True                    # 是否打印调整信息
        
        # 自定义配置覆盖 - 预定义所有可能的损失类型
        self._C.LOSS_SCHEDULER.OVERRIDES = CN()
        
        # 为每种损失类型预定义配置结构
        for loss_type in ['mse', 'ssim', 'edge', 'gradient', 'boundary', 'transparency', 'perceptual', 'histogram']:
            self._C.LOSS_SCHEDULER.OVERRIDES[loss_type] = CN()
            self._C.LOSS_SCHEDULER.OVERRIDES[loss_type].schedule_type = ""
            self._C.LOSS_SCHEDULER.OVERRIDES[loss_type].initial_weight = 0.0
            self._C.LOSS_SCHEDULER.OVERRIDES[loss_type].final_weight = 0.0
            self._C.LOSS_SCHEDULER.OVERRIDES[loss_type].start_epoch = 1
            self._C.LOSS_SCHEDULER.OVERRIDES[loss_type].warmup_epochs = 10
            self._C.LOSS_SCHEDULER.OVERRIDES[loss_type].decay_rate = 0.95
            self._C.LOSS_SCHEDULER.OVERRIDES[loss_type].step_size = 30
            self._C.LOSS_SCHEDULER.OVERRIDES[loss_type].step_gamma = 0.5

        self._C.TESTING = CN()
        self._C.TESTING.WEIGHT = None
        self._C.TESTING.SAVE_IMAGES = True
        self._C.TESTING.BATCH_SIZE = 4
        self._C.TESTING.NUM_WORKERS = 4


        # Override parameter values from YAML file first, then from override list.
        self._C.merge_from_file(config_yaml)
        self._C.merge_from_list(config_override)

        # Make an instantiated object of this class immutable.
        self._C.freeze()

    def dump(self, file_path: str):
        r"""Save config at the specified file path.

        Parameters
        ----------
        file_path: str
            (YAML) path to save config at.
        """
        self._C.dump(stream=open(file_path, "w"))

    def __getattr__(self, attr: str):
        return self._C.__getattr__(attr)

    def get_loss_config(self):
        """
        获取当前选择的损失配置
        
        Returns:
            dict: 包含损失权重的字典
        """
        loss_type = self._C.LOSS.TYPE
        if hasattr(self._C.LOSS.CONFIGS, loss_type):
            config = getattr(self._C.LOSS.CONFIGS, loss_type)
            loss_config = {
                'mse_weight': getattr(config, 'MSE_WEIGHT', 1.0),
                'ssim_weight': config.SSIM_WEIGHT,
                'edge_weight': config.EDGE_WEIGHT,
                'gradient_weight': config.GRADIENT_WEIGHT,
                'boundary_weight': config.BOUNDARY_WEIGHT,
                'transparency_weight': config.TRANSPARENCY_WEIGHT,
                'fade_width': config.FADE_WIDTH,
                'min_alpha': config.MIN_ALPHA
            }
            
            # 添加新的损失权重（如果存在）
            if hasattr(config, 'PERCEPTUAL_WEIGHT'):
                loss_config['perceptual_weight'] = config.PERCEPTUAL_WEIGHT
            else:
                loss_config['perceptual_weight'] = 0.05  # 默认值
                
            if hasattr(config, 'HISTOGRAM_WEIGHT'):
                loss_config['histogram_weight'] = config.HISTOGRAM_WEIGHT
            else:
                loss_config['histogram_weight'] = 0.02  # 默认值
                
            return loss_config
        else:
            # 如果指定的配置不存在，返回默认配置
            print(f"警告: 损失配置 '{loss_type}' 不存在，使用默认配置")
            config = self._C.LOSS.CONFIGS.DEFAULT
            return {
                'mse_weight': getattr(config, 'MSE_WEIGHT', 1.0),
                'ssim_weight': config.SSIM_WEIGHT,
                'edge_weight': config.EDGE_WEIGHT,
                'gradient_weight': config.GRADIENT_WEIGHT,
                'boundary_weight': config.BOUNDARY_WEIGHT,
                'transparency_weight': config.TRANSPARENCY_WEIGHT,
                'perceptual_weight': getattr(config, 'PERCEPTUAL_WEIGHT', 0.05),
                'histogram_weight': getattr(config, 'HISTOGRAM_WEIGHT', 0.02),
                'fade_width': config.FADE_WIDTH,
                'min_alpha': config.MIN_ALPHA
            }

    def __repr__(self):
        return self._C.__repr__()
