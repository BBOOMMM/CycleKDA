import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

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
MODEL_DTYPE = torch.float32


class SequenceDataset(Dataset):
    def __init__(self, features, labels, indices, time_stride=3):
        self.features = features
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.time_stride = time_stride

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        x = self.features[real_idx, :: self.time_stride, :]
        y = self.labels[real_idx, :: self.time_stride, :]
        return {
            "inputs": torch.from_numpy(np.array(x, dtype=np.float32, copy=True)),
            "labels": torch.from_numpy(np.array(y, dtype=np.float32, copy=True)),
        }


class LSTMTimeModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        out, _ = self.lstm(inputs)
        preds = self.head(out)
        return preds


def parse_args():
    parser = argparse.ArgumentParser(description="LSTM inference")
    parser.add_argument("--data-path", type=str, default="/mnt/nvme2/chenxuanyu/minv2_exp")
    parser.add_argument("--features-file", type=str, default="features.npy")
    parser.add_argument("--labels-file", type=str, default="labels.npy")
    parser.add_argument("--indexes-file", type=str, default="index.h5")
    parser.add_argument("--mmap-mode", type=str, default="r")
    parser.add_argument("--time-stride", type=int, default=3)

    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--ckpt-dir", type=str, default="lstm_ckpt")
    parser.add_argument("--weights-file", type=str, default="pytorch_model.bin")
    parser.add_argument("--max-test-samples", type=int, default=0, help="0 means all test samples")

    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/chenxuanyu/code/CycleKDA/pred2",
        help="Directory for inference outputs.",
    )
    parser.add_argument("--pred-out", type=str, default="lstm_test_pred.npy")
    return parser.parse_args()


def load_model(args, input_size, output_size):
    model = LSTMTimeModel(
        input_size=input_size,
        hidden_size=args.hidden_size,
        output_size=output_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(DEVICE).to(MODEL_DTYPE)

    weight_path = os.path.join(args.ckpt_dir, args.weights_file)
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"weights not found: {weight_path}")

    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _restore_time_length(preds_np, target_len, time_stride):
    if time_stride <= 1:
        if preds_np.shape[1] != target_len:
            raise ValueError(f"Prediction length {preds_np.shape[1]} does not match target length {target_len}.")
        return preds_np

    restored = np.repeat(preds_np, time_stride, axis=1)
    if restored.shape[1] < target_len:
        pad_len = target_len - restored.shape[1]
        tail = np.repeat(restored[:, -1:, :], pad_len, axis=1)
        restored = np.concatenate([restored, tail], axis=1)
    elif restored.shape[1] > target_len:
        restored = restored[:, :target_len, :]
    return restored


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

    dataset = SequenceDataset(
        features=features,
        labels=labels,
        indices=test_idx,
        time_stride=args.time_stride,
    )
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
    print(f"test samples: {len(dataset)}, input_size: {input_size}, output_size: {output_size}, pred_len: {pred_len}")
    output_dir = args.output_dir.strip() if args.output_dir.strip() else args.data_path
    os.makedirs(output_dir, exist_ok=True)
    pred_path = os.path.join(output_dir, args.pred_out)
    print(f"writing predictions to: {pred_path}")

    try:
        pred_memmap = np.lib.format.open_memmap(
            pred_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(dataset), pred_len, output_size),
        )
    except PermissionError as e:
        raise PermissionError(
            f"Cannot write prediction file to '{pred_path}'. "
            "Use --output-dir to point to a writable directory."
        ) from e

    write_pos = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = batch["inputs"].to(DEVICE, dtype=MODEL_DTYPE)

            preds = model(inputs)
            preds_np = preds.detach().to(torch.float32).cpu().numpy()
            preds_np = _restore_time_length(preds_np, target_len=pred_len, time_stride=args.time_stride)

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
