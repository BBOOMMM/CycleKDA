import os
import numpy as np
import pandas as pd


def load_data(
    data_path,
    features_file='features.npy',
    labels_file='labels.npy',
    indexes_file='index.h5',
    mmap_mode='r',
):
    features_path = os.path.join(data_path, features_file)
    labels_path = os.path.join(data_path, labels_file)
    indexes_path = os.path.join(data_path, indexes_file)

    features = np.load(features_path, mmap_mode=mmap_mode)
    labels = np.load(labels_path, mmap_mode=mmap_mode)
    indexes = pd.read_hdf(indexes_path)
    indexes['datetime'] = pd.to_datetime(indexes['datetime'])
    return features, labels, indexes


def split_data(
    data_path,
    features_file='features.npy',
    labels_file='labels.npy',
    indexes_file='index.h5',
    mmap_mode='r',
    train_start='2024-07-01',
    train_end='2024-11-30 23:59:59',
    test_start='2024-12-01',
    test_end='2024-12-31 23:59:59',
    materialize=False,
):
    features, labels, indexes = load_data(
        data_path=data_path,
        features_file=features_file,
        labels_file=labels_file,
        indexes_file=indexes_file,
        mmap_mode=mmap_mode,
    )

    train_start = pd.Timestamp(train_start)
    train_end = pd.Timestamp(train_end)
    test_start = pd.Timestamp(test_start)
    test_end = pd.Timestamp(test_end)

    dt = indexes['datetime']
    train_mask = (dt >= train_start) & (dt <= train_end)
    test_mask = (dt >= test_start) & (dt <= test_end)

    train_idx = np.where(train_mask.to_numpy())[0]
    test_idx = np.where(test_mask.to_numpy())[0]

    if materialize:
        # Advanced indexing on memmap creates new in-memory arrays.
        train_features = features[train_idx]
        train_labels = labels[train_idx]
        test_features = features[test_idx]
        test_labels = labels[test_idx]
        return train_features, train_labels, test_features, test_labels

    # Low-RAM path: keep memmap arrays and pass lightweight indices downstream.
    return features, labels, train_idx, test_idx


if __name__ == '__main__':
    data_path = '/mnt/nvme2/chenxuanyu/minv2_exp/'
    features, labels, train_idx, test_idx = split_data(data_path)
    print('--- split summary ---')
    print(f'train rows: {len(train_idx)}')
    print(f'test rows: {len(test_idx)}')
    print('full features shape:', features.shape, 'full labels shape:', labels.shape)
