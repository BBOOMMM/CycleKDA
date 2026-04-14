import argparse
import json
import os

import numpy as np
import pandas as pd


def parse_args():
	parser = argparse.ArgumentParser(
		description="Aggregate per-channel IC/RankIC/IR means across dates from evaluate outputs."
	)
	parser.add_argument(
		"--pred-root",
		type=str,
		default="/home/chenxuanyu/code/CycleKDA/pred2",
		help="Root directory containing evaluation subfolders, e.g. baseline/ and cyclekda/.",
	)
	parser.add_argument(
		"--targets",
		type=str,
		default="cyclekda",
		help="Comma-separated evaluation folder names under pred-root.",
	)
	parser.add_argument(
		"--output-name",
		type=str,
		default="channel_mean_metrics",
		help="Base filename for output csv/json.",
	)
	return parser.parse_args()


def load_metric_arrays(folder, prefix):
	ic_path = os.path.join(folder, f"{prefix}_ic.npy")
	rank_ic_path = os.path.join(folder, f"{prefix}_rank_ic.npy")
	ir_path = os.path.join(folder, f"{prefix}_ir.npy")

	if not os.path.exists(ic_path):
		raise FileNotFoundError(f"Missing file: {ic_path}")
	if not os.path.exists(rank_ic_path):
		raise FileNotFoundError(f"Missing file: {rank_ic_path}")
	if not os.path.exists(ir_path):
		raise FileNotFoundError(f"Missing file: {ir_path}")

	ic = np.load(ic_path)
	rank_ic = np.load(rank_ic_path)
	ir = np.load(ir_path)
	return ic, rank_ic, ir


def summarize_per_channel(ic, rank_ic, ir):
	if ic.shape != rank_ic.shape or ic.shape != ir.shape:
		raise ValueError(
			f"Shape mismatch: ic={ic.shape}, rank_ic={rank_ic.shape}, ir={ir.shape}"
		)

	num_dates, num_channels = ic.shape
	rows = []
	for c in range(num_channels):
		ic_c = ic[:, c]
		rank_ic_c = rank_ic[:, c]
		ir_c = ir[:, c]
		rows.append(
			{
				"output_channel": c,
				"mean_ic": float(np.mean(ic_c[np.isfinite(ic_c)])) if np.isfinite(ic_c).any() else float("nan"),
				"mean_rank_ic": float(np.mean(rank_ic_c[np.isfinite(rank_ic_c)])) if np.isfinite(rank_ic_c).any() else float("nan"),
				"mean_ir": float(np.mean(ir_c[np.isfinite(ir_c)])) if np.isfinite(ir_c).any() else float("nan"),
			}
		)
	return rows, num_dates, num_channels


def main():
	args = parse_args()
	targets = [x.strip() for x in args.targets.split(",") if x.strip()]

	for name in targets:
		folder = os.path.join(args.pred_root, name)
		if not os.path.isdir(folder):
			print(f"skip: folder not found -> {folder}")
			continue

		ic, rank_ic, ir = load_metric_arrays(folder, name)
		rows, num_dates, num_channels = summarize_per_channel(ic, rank_ic, ir)

		df = pd.DataFrame(rows)
		csv_path = os.path.join(folder, f"{args.output_name}.csv")
		df.to_csv(csv_path, index=False)

		summary = {
			"target": name,
			"num_dates": int(num_dates),
			"num_channels": int(num_channels),
			"rows": rows,
		}
		json_path = os.path.join(folder, f"{args.output_name}.json")
		with open(json_path, "w", encoding="utf-8") as f:
			json.dump(summary, f, ensure_ascii=False, indent=2)

		print(f"[{name}] num_dates={num_dates}, num_channels={num_channels}")
		print(df.to_string(index=False))
		print(f"saved: {csv_path}")
		print(f"saved: {json_path}")


if __name__ == "__main__":
	main()
