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


def analyze_indexes(data_path, indexes_file='index.h5'):
    indexes_path = os.path.join(data_path, indexes_file)
    indexes = pd.read_hdf(indexes_path)
    indexes['datetime'] = pd.to_datetime(indexes['datetime'])

    print(f'total rows: {len(indexes)}')
    print(f'columns: {indexes.columns.tolist()}')
    print(f'start datetime: {indexes["datetime"].min()}')
    print(f'end datetime: {indexes["datetime"].max()}')
    print()

    # 1. 查看总共有多少个不同日期, 多少个不同股票
    date_counts = indexes['datetime'].value_counts().sort_index()
    print(f'unique dates: {len(date_counts)}')
    code_counts = indexes['code'].value_counts().sort_index()
    print(f'unique codes: {len(code_counts)}')
    print()

    # 2. 按日期分组，查看每天有多少张股票
    daily_code_counts = indexes.groupby('datetime')['code'].count()   # .nunique() 计算每个日期的不同股票数量
    print('daily code counts:')
    print(daily_code_counts.describe())
    daily_unique_code_counts = indexes.groupby('datetime')['code'].nunique()   # .nunique() 计算每个日期的不同股票数量
    print('daily unique code counts:')
    print(daily_unique_code_counts.describe())
    # 两个一摸一样，按日期分组后，组内所有的股票都是不同的


if __name__ == '__main__':
    data_path = '/mnt/nvme2/chenxuanyu/minv2_exp/'
    # features, labels, train_idx, test_idx = split_data(data_path)
    # print('--- split summary ---')
    # print(f'train rows: {len(train_idx)}')
    # print(f'test rows: {len(test_idx)}')
    # print('full features shape:', features.shape, 'full labels shape:', labels.shape)
    analyze_indexes(data_path)