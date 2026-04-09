import numpy as np
import os
import pandas as pd

data_path = '/mnt/nvme2/chenxuanyu/minv2_exp/'
features_filename = 'features.npy'
features_path = os.path.join(data_path, features_filename)
labels_filename = 'labels.npy'
labels_path = os.path.join(data_path, labels_filename)
indexes_filename = 'index.h5'
indexes_path = os.path.join(data_path, indexes_filename)

features = np.load(features_path, mmap_mode="r")
labels = np.load(labels_path, mmap_mode="r")

print(features.shape, features.dtype)
# (621227, 711, 33) float32
print(labels.shape, labels.dtype)
# (621227, 711, 10) float64

indexes = pd.read_hdf(indexes_path)
print(indexes.tail(5))
indexes['datetime'] = pd.to_datetime(indexes['datetime'])
print(indexes['datetime'].min(), indexes['datetime'].max())