import os
import random
from collections import OrderedDict

import numpy as np
import torch


def seed_everything(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, epoch, model_name, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    checkpoint_file = os.path.join(outdir, model_name + '_' + 'epoch_' + str(epoch) + '.pth')
    torch.save(state, checkpoint_file)


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights, map_location=lambda storage, loc: storage.cuda(0))
    new_state_dict = OrderedDict()
    for key, value in checkpoint['state_dict'].items():
        if key.startswith('module'):
            name = key[7:]
        else:
            name = key
        new_state_dict[name] = value
    
    # 兼容性加载：如果权重中没有边界注意力模块，但模型有，则跳过这些层
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(new_state_dict.keys())
    
    # 找出模型中有但权重中没有的键（新增的边界注意力模块）
    missing_keys = model_keys - checkpoint_keys
    # 找出权重中有但模型中没有的键
    unexpected_keys = checkpoint_keys - model_keys
    
    if missing_keys:
        print(f"警告: 权重文件中缺少以下键（可能是新增的边界注意力模块）:")
        for key in sorted(missing_keys):
            if 'doc_boundary' in key:
                print(f"  - {key}")
        print("这些层将使用随机初始化权重。")
    
    if unexpected_keys:
        print(f"警告: 权重文件中有以下多余的键:")
        for key in sorted(unexpected_keys):
            print(f"  - {key}")
    
    # 只加载匹配的权重
    filtered_state_dict = {k: v for k, v in new_state_dict.items() if k in model_keys}
    
    # 使用strict=False来允许部分加载
    model.load_state_dict(filtered_state_dict, strict=False)
    
    print(f"成功加载 {len(filtered_state_dict)}/{len(model_keys)} 个权重参数")

