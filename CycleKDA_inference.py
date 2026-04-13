import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from KimiLinear import KimiLinearConfig, KimiLinearTimeModel
from load_data import split_data


def set_seed():
	import random
	from transformers import set_seed as hf_set_seed

	seed = 42
	os.environ["PYTHONHASHSEED"] = str(seed)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	hf_set_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


set_seed()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


class CycleKDADataset(Dataset):
	def __init__(self, features, labels, indices):
		self.features = features
		self.labels = labels
		self.indices = np.asarray(indices, dtype=np.int64)

	def __len__(self):
		return len(self.indices)

	def __getitem__(self, idx):
		real_idx = int(self.indices[idx])
		x = self.features[real_idx, :, :]
		y = self.labels[real_idx, :, :]
		return {
			"inputs": torch.from_numpy(np.array(x, dtype=np.float32, copy=True)),
			"labels": torch.from_numpy(np.array(y, dtype=np.float32, copy=True)),
		}


def parse_args():
	parser = argparse.ArgumentParser(description="CycleKDA inference")
	parser.add_argument("--data-path", type=str, default="/mnt/nvme2/chenxuanyu/minv2_exp")
	parser.add_argument("--features-file", type=str, default="features.npy")
	parser.add_argument("--labels-file", type=str, default="labels.npy")
	parser.add_argument("--indexes-file", type=str, default="index.h5")
	parser.add_argument("--mmap-mode", type=str, default="r")
	parser.add_argument("--T_cycle", type=int, default=3)
	parser.add_argument("--batch-size", type=int, default=64)
	parser.add_argument("--num-workers", type=int, default=0)
	parser.add_argument("--ckpt-dir", type=str, default="CycleKDA_ckpt")
	parser.add_argument("--weights-file", type=str, default="pytorch_model.bin")
	parser.add_argument("--max-test-samples", type=int, default=0, help="0 means all test samples")
	parser.add_argument("--output-dir", type=str, default="/home/chenxuanyu/code/CycleKDA/pred")
	parser.add_argument("--pred-out", type=str, default="cyclekda_test_pred.npy")
	parser.add_argument("--metrics-out", type=str, default="cyclekda_test_metrics.json")
	return parser.parse_args()


def load_model(args, input_size, output_size):
	cfg_path = os.path.join(os.path.dirname(__file__), "configs/timekimi_config.json")
	with open(cfg_path, "r", encoding="utf-8") as f:
		cfg_json = json.load(f)

	cfg_json["input_size"] = input_size
	cfg_json["output_size"] = output_size
	config = KimiLinearConfig(**cfg_json)
	config.linear_attn_config["T_cycle"] = args.T_cycle

	model = KimiLinearTimeModel(config).to(DEVICE).to(MODEL_DTYPE)

	weight_path = os.path.join(args.ckpt_dir, args.weights_file)
	if not os.path.exists(weight_path):
		raise FileNotFoundError(f"weights not found: {weight_path}")

	state_dict = torch.load(weight_path, map_location="cpu")
	model.load_state_dict(state_dict, strict=True)
	model.eval()
	return model


def evaluate_and_save(args):
	features, labels, _, test_idx = split_data(
		data_path=args.data_path,
		features_file=args.features_file,
		labels_file=args.labels_file,
		indexes_file=args.indexes_file,
		mmap_mode=args.mmap_mode,
		materialize=False,
	)

	if args.max_test_samples > 0:
		test_idx = test_idx[: args.max_test_samples]

	dataset = CycleKDADataset(features=features, labels=labels, indices=test_idx)
	dataloader = DataLoader(
		dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=torch.cuda.is_available(),
	)

	input_size = features.shape[-1]
	output_size = labels.shape[-1]
	model = load_model(args, input_size=input_size, output_size=output_size)

	pred_len = features.shape[1]
	output_dir = args.output_dir.strip() if args.output_dir.strip() else args.data_path
	os.makedirs(output_dir, exist_ok=True)
	pred_path = os.path.join(output_dir, args.pred_out)

	pred_memmap = np.lib.format.open_memmap(
		pred_path,
		mode="w+",
		dtype=np.float32,
		shape=(len(dataset), pred_len, output_size),
	)

	write_pos = 0

	with torch.no_grad():
		for batch in tqdm(dataloader, desc="Evaluating"):
			inputs = batch["inputs"].to(DEVICE, dtype=MODEL_DTYPE)
			targets = batch["labels"].to(DEVICE, dtype=MODEL_DTYPE)

			preds, _ = model(input_ids=inputs, use_cache=False)
			preds_np = preds.detach().to(torch.float32).cpu().numpy()
			if preds_np.shape[1] != pred_len:
				raise ValueError(
					f"Prediction length mismatch: got {preds_np.shape[1]}, expected {pred_len}."
				)

			bs = preds_np.shape[0]
			pred_memmap[write_pos : write_pos + bs] = preds_np
			write_pos += bs

	pred_memmap.flush()

	print(f"device: {DEVICE}")
	print(f"test samples: {len(dataset)}")
	print(f"saved predictions: {pred_path}")


def main():
	args = parse_args()
	evaluate_and_save(args)


if __name__ == "__main__":
	main()
