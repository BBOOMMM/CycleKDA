import argparse
import json
import os

import numpy as np
import pandas as pd

from load_data import load_data, split_data, labels_normalize
from tqdm import tqdm


def _pearson_corr_1d(x, y):
	x = np.asarray(x, dtype=np.float64)
	y = np.asarray(y, dtype=np.float64)
	valid = np.isfinite(x) & np.isfinite(y)
	if valid.sum() < 2:
		return np.nan
	x = x[valid]
	y = y[valid]
	x = x - x.mean()
	y = y - y.mean()
	denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
	if denom <= 0:
		return np.nan
	return float(np.sum(x * y) / denom)


def _rankdata_1d(x):
	order = np.argsort(x, kind="mergesort")
	ranks = np.empty_like(order, dtype=np.float64)
	ranks[order] = np.arange(len(x), dtype=np.float64)
	return ranks


def _corr_series_across_time(pred_day, label_day, channel_idx):
	seq_len = pred_day.shape[1]
	ic_list = []
	rank_ic_list = []
	for t in range(seq_len):
		p = pred_day[:, t, channel_idx]
		y = label_day[:, t, channel_idx]
		ic_val = _pearson_corr_1d(p, y)
		rank_ic_val = _pearson_corr_1d(_rankdata_1d(p), _rankdata_1d(y))
		ic_list.append(ic_val)
		rank_ic_list.append(rank_ic_val)
	return np.asarray(ic_list, dtype=np.float64), np.asarray(rank_ic_list, dtype=np.float64)


def parse_args():
	parser = argparse.ArgumentParser(description="Evaluate predictions by date x output_channel")
	parser.add_argument("--data-path", type=str, default="/mnt/nvme2/chenxuanyu/minv2_exp")
	parser.add_argument("--features-file", type=str, default="features.npy")
	parser.add_argument("--labels-file", type=str, default="labels.npy")
	parser.add_argument("--indexes-file", type=str, default="index.h5")
	parser.add_argument("--mmap-mode", type=str, default="r")
	parser.add_argument("--pred-path", type=str, default="/home/chenxuanyu/code/CycleKDA/pred2/lstm_test_pred.npy")
	parser.add_argument("--output-dir", type=str, default="/home/chenxuanyu/code/CycleKDA/pred2/lstm")
	parser.add_argument("--prefix", type=str, default="lstm")
	parser.add_argument("--max-test-samples", type=int, default=0, help="0 means all test samples")
	return parser.parse_args()


def main():
	args = parse_args()

	_, labels, indexes = load_data(
		data_path=args.data_path,
		features_file=args.features_file,
		labels_file=args.labels_file,
		indexes_file=args.indexes_file,
		mmap_mode=args.mmap_mode,
	)
	# labels = labels_normalize(labels)
	_, _, _, test_idx = split_data(
		data_path=args.data_path,
		features_file=args.features_file,
		labels_file=args.labels_file,
		indexes_file=args.indexes_file,
		mmap_mode=args.mmap_mode,
		materialize=False,
	)

	if args.max_test_samples > 0:
		test_idx = test_idx[: args.max_test_samples]

	pred = np.load(args.pred_path, mmap_mode="r")
	if pred.shape[0] != len(test_idx):
		raise ValueError(f"Prediction row count mismatch: pred has {pred.shape[0]} rows, test_idx has {len(test_idx)} rows.")

	test_indexes = indexes.iloc[test_idx].copy().reset_index(drop=True)
	test_indexes["date"] = test_indexes["datetime"].dt.date
	assert pred.shape[0] == test_indexes.shape[0], f"Row count mismatch: pred has {pred.shape[0]} rows, test_indexes has {test_indexes.shape[0]} rows."

	unique_dates = np.sort(test_indexes["date"].unique())
	num_dates = len(unique_dates)
	num_channels = pred.shape[2]

	ic_daily = np.full((num_dates, num_channels), np.nan, dtype=np.float64)
	rank_ic_daily = np.full((num_dates, num_channels), np.nan, dtype=np.float64)
	ir_daily = np.full((num_dates, num_channels), np.nan, dtype=np.float64)

	for d_i, day in tqdm(enumerate(unique_dates), total=len(unique_dates), desc="Evaluating dates"):
		day_mask = (test_indexes["date"] == day).to_numpy()
		day_pos = np.where(day_mask)[0]

		pred_day = pred[day_pos]  # [num_codes_day, L, C]
		label_day = labels[test_idx[day_pos]]

		if pred_day.shape[1] != label_day.shape[1] or pred_day.shape[2] != label_day.shape[2]:
			raise ValueError(f"Shape mismatch on {day}: pred {pred_day.shape}, label {label_day.shape}.")

		for c in range(num_channels):
			ic_series, rank_ic_series = _corr_series_across_time(pred_day, label_day, c)

			ic_valid = np.isfinite(ic_series)
			rank_ic_valid = np.isfinite(rank_ic_series)
			ic_mean = np.mean(ic_series[ic_valid]) if ic_valid.any() else np.nan
			rank_ic_mean = np.mean(rank_ic_series[rank_ic_valid]) if rank_ic_valid.any() else np.nan
			ic_std = np.std(ic_series[ic_valid], ddof=0) if ic_valid.any() else np.nan
			ir_val = ic_mean / ic_std if np.isfinite(ic_std) and ic_std > 0 else np.nan

			ic_daily[d_i, c] = ic_mean
			rank_ic_daily[d_i, c] = rank_ic_mean
			ir_daily[d_i, c] = ir_val

	os.makedirs(args.output_dir, exist_ok=True)

	ic_path = os.path.join(args.output_dir, f"{args.prefix}_ic.npy")
	rank_ic_path = os.path.join(args.output_dir, f"{args.prefix}_rank_ic.npy")
	ir_path = os.path.join(args.output_dir, f"{args.prefix}_ir.npy")
	np.save(ic_path, ic_daily)
	np.save(rank_ic_path, rank_ic_daily)
	np.save(ir_path, ir_daily)

	records = []
	for d_i, day in enumerate(unique_dates):
		for c in range(num_channels):
			records.append(
				{
					"date": str(day),
					"output_channel": int(c),
					"ic": float(ic_daily[d_i, c]),
					"rank_ic": float(rank_ic_daily[d_i, c]),
					"ir": float(ir_daily[d_i, c]),
				}
			)
	detail_df = pd.DataFrame(records)
	detail_csv_path = os.path.join(args.output_dir, f"{args.prefix}_detail.csv")
	detail_df.to_csv(detail_csv_path, index=False)

	summary = {
		"num_dates": int(num_dates),
		"num_channels": int(num_channels),
		"num_metrics": 3,
		"total_numbers": int(num_dates * num_channels * 3),
		"ic_path": ic_path,
		"rank_ic_path": rank_ic_path,
		"ir_path": ir_path,
		"detail_csv_path": detail_csv_path,
		"global_ic_mean": float(np.mean(ic_daily[np.isfinite(ic_daily)])) if np.isfinite(ic_daily).any() else float("nan"),
		"global_rank_ic_mean": float(np.mean(rank_ic_daily[np.isfinite(rank_ic_daily)])) if np.isfinite(rank_ic_daily).any() else float("nan"),
		"global_ir_mean": float(np.mean(ir_daily[np.isfinite(ir_daily)])) if np.isfinite(ir_daily).any() else float("nan"),
	}
	summary_path = os.path.join(args.output_dir, f"{args.prefix}_summary.json")
	with open(summary_path, "w", encoding="utf-8") as f:
		json.dump(summary, f, ensure_ascii=False, indent=2)

	print(f"num_dates={num_dates}, num_channels={num_channels}, total_numbers={summary['total_numbers']}")
	print(f"saved: {ic_path}")
	print(f"saved: {rank_ic_path}")
	print(f"saved: {ir_path}")
	print(f"saved: {detail_csv_path}")
	print(f"saved: {summary_path}")


if __name__ == "__main__":
	main()