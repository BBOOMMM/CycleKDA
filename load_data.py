import numpy as np
import pandas as pd

df = pd.read_hdf('/mnt/nvme2/chenxuanyu/minv2_exp/index.h5')
print(df)

features = np.load('/mnt/nvme2/chenxuanyu/minv2_exp/features.npy', mmap_mode='r')
print(features.shape)