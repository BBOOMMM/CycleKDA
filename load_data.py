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


def analyze_features_labels(data_path, features_file='features.npy', labels_file='labels.npy', mmap_mode='r'):
    features_path = os.path.join(data_path, features_file)
    labels_path = os.path.join(data_path, labels_file)

    features = np.load(features_path, mmap_mode=mmap_mode)
    labels = np.load(labels_path, mmap_mode=mmap_mode)

    # 均值和标准差
    print('features mean:', np.mean(features, axis=(0, 1)))
    print('features std:', np.std(features, axis=(0, 1)))
    print('labels mean:', np.mean(labels, axis=(0, 1)))
    print('labels std:', np.std(labels, axis=(0, 1)))

    # features mean: [ 0.2206602   0.08436979  0.16428365  0.14314     0.10973248  0.08810733
    #     0.03798394  0.60774297  0.60774297  0.416209    0.36705777  0.03798394
    #     -0.00320224  0.01000556  0.0132598   0.01024407  0.01085689  0.03798394
    #     0.03798394  0.02013082  0.02183997  0.03798394  0.03798394  0.02143955
    #     0.02154153  0.03798394  0.03798394  0.01002709  0.01448373  0.03798394
    #     0.03798394  0.01059811  0.01468828]
    # features std: [2.3063872  0.4228932  0.6744914  0.6139998  0.5806322  0.4787265
    #     0.15173794 2.2049816  2.1738532  1.5496777  1.4419651  0.19489467
    #     0.09611553 0.06632112 0.07933424 0.03784547 0.04157137 0.19489467
    #     0.17343128 0.07465836 0.09896561 0.19489467 0.18870108 0.0793663
    #     0.08383376 0.19489467 0.13986966 0.03693069 0.06059893 0.19489467
    #     0.14036547 0.04065445 0.05351375]
    # labels mean: [4.35800351e-07 1.20894381e-06 3.93306257e-06 1.38019257e-05
    #     2.20188773e-05 3.76767269e-05 5.90833497e-05 8.26720135e-05
    #     1.83049592e-04 8.83175792e-05]
    # labels std: [0.00086206 0.00146025 0.00204819 0.00337449 0.0042574  0.00576091
    #     0.00766303 0.00905497 0.0118228  0.01478735]


def scale_labels(
    labels,
    target_std=0.5,
    std_range=(0.1, 1.0),
    value_range=(-1.0, 1.0),
    eps=1e-8,
):
    """
    Pure linear scaling for labels (no tanh/clip/nonlinear transform).
    Per-channel de-mean + standardize, then apply one global scale alpha.

    alpha is chosen as min(alpha_for_target_std, alpha_for_value_range), so the
    output stays inside value_range without nonlinear truncation.
    """

    lo, hi = value_range
    min_std, max_std = std_range

    orig_mean = np.mean(labels, axis=(0, 1))
    orig_std = np.std(labels, axis=(0, 1))

    centered = labels - orig_mean[None, None, :]
    safe_std = np.where(orig_std > eps, orig_std, 1.0)
    z = centered / safe_std[None, None, :]

    # For standardized z, per-channel std is near 1, so alpha targets std directly.
    alpha_for_target_std = target_std

    max_abs_z = float(np.max(np.abs(z)))
    if max_abs_z > eps:
        range_abs = min(abs(lo), abs(hi))
        alpha_for_value_range = range_abs / max_abs_z
    else:
        alpha_for_value_range = alpha_for_target_std

    alpha = float(min(alpha_for_target_std, alpha_for_value_range))
    labels_scaled = z * alpha

    scaled_mean = np.mean(labels_scaled, axis=(0, 1))
    scaled_std = np.std(labels_scaled, axis=(0, 1))

    print('labels mean (before):', orig_mean)
    print('labels std (before):', orig_std)
    print('scale alpha:', alpha)
    print('labels mean (after):', scaled_mean)
    print('labels std (after):', scaled_std)
    print('labels min/max (after):', float(np.min(labels_scaled)), float(np.max(labels_scaled)))

    out_of_std_range = np.where((scaled_std < min_std) | (scaled_std > max_std))[0]
    if len(out_of_std_range) > 0:
        print(
            'warning: some channels std out of requested range after clipping:',
            out_of_std_range.tolist(),
        )

    return labels_scaled



def labels_normalize(labels, eps=1e-8):
    if labels.ndim != 3:
        raise ValueError(f'Expected labels with shape [N, L, C], got {labels.shape}')

    # labels = np.asarray(labels, dtype=np.float32)
    # mean = np.mean(labels, axis=(0, 1))
    # std = np.std(labels, axis=(0, 1))
    # safe_std = np.where(std > eps, std, 1.0)
    # normalized = (labels - mean[None, None, :]) / safe_std[None, None, :]

    normalized_labels = (labels - labels.min(axis=(0, 1), keepdims=True)) / (labels.max(axis=(0, 1), keepdims=True) - labels.min(axis=(0, 1), keepdims=True) + eps)
    normalized_labels = normalized_labels * 100 - 50  # scale to [-50, 50]
    # print('labels mean (after):', np.mean(normalized_labels, axis=(0, 1)))
    # print('labels std (after):', np.std(normalized_labels, axis=(0, 1)))
    # labels mean (after): [-14.70492752  -1.79464115   5.53005659   3.13515648   3.58532298
    #     -3.10477326  -7.91713323 -17.15840575 -16.70718168 -21.54044149]
    # labels std (after): [0.20688712 0.26089737 0.24113006 0.27956522 0.35569515 0.42120786
    #     0.51684869 0.57447077 0.62858135 0.77543044]

    return normalized_labels



if __name__ == '__main__':
    data_path = '/mnt/nvme2/chenxuanyu/minv2_exp/'
    
    # features, labels, train_idx, test_idx = split_data(data_path)
    # print('--- split summary ---')
    # print(f'train rows: {len(train_idx)}')
    # print(f'test rows: {len(test_idx)}')
    # print('full features shape:', features.shape, 'full labels shape:', labels.shape)

    # analyze_indexes(data_path)

    # analyze_features_labels(data_path)
    
    _, labels, _ = load_data(data_path)
    # labels_scaled = scale_labels(labels)
    labels_normalized = labels_normalize(labels)