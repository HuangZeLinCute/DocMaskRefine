import os
from .dataset_RGB import DataReader


def get_data(img_dir, inp, tar, mode='train', ori=False, img_options=None):
    print("DEBUG >>> img_dir =", img_dir)
    assert os.path.exists(img_dir)
    return DataReader(img_dir, inp, tar, mode, ori, img_options)

