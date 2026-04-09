import numpy as np
import os
import torch
import argparse
from torch.utils.data import DataLoader, Dataset
import json
from KimiLinear import KimiLinearConfig, KimiLinearTimeModel
import torch.nn as nn
from tqdm import tqdm 
import matplotlib.pyplot as plt
import math
from load_data import split_data


def set_seed():
    import random
    import numpy as np
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


def parse_args():
    parser = argparse.ArgumentParser(description="CycleKDA")
    parser.add_argument("--data-path", type=str, default="/mnt/nvme2/chenxuanyu/minv2_exp")
    parser.add_argument("--features-file", type=str, default="features.npy")
    parser.add_argument("--labels-file", type=str, default="labels.npy")
    parser.add_argument("--indexes-file", type=str, default="index.h5")
    parser.add_argument("--mmap-mode", type=str, default="r")
    # parser.add_argument(
    #     "--time-stride",
    #     type=int,
    #     default=3,
    #     help="Baseline uses features[:, :, ::time_stride] and labels[:, :, ::time_stride]",
    # )
    parser.add_argument(
        "--T_cycle",
        type=int,
        default=3,
        help="various values",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument(
        "--disable-data-parallel",
        action="store_true",
        help="Disable torch.nn.DataParallel even when multiple GPUs are visible.",
    )
    parser.add_argument(
        "--device-ids",
        type=str,
        default="",
        help="Comma-separated CUDA device ids for DataParallel, e.g. '0,1,2'. Empty means all visible devices.",
    )
    return parser.parse_args()


def load_train_data(args):
    features, labels, train_idx, test_idx = split_data(
        data_path=args.data_path,
        features_file=args.features_file,
        labels_file=args.labels_file,
        indexes_file=args.indexes_file,
        mmap_mode=args.mmap_mode,
        materialize=False,
    )
    return features, labels, train_idx, test_idx


class BaselineDataset(Dataset):
    def __init__(self, features, labels, indices=None):
        self.features = features
        self.labels = labels
        self.indices = None if indices is None else np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return self.features.shape[0] if self.indices is None else len(self.indices)

    def __getitem__(self, idx):
        real_idx = idx if self.indices is None else int(self.indices[idx])
        x = self.features[real_idx, :, :]
        y = self.labels[real_idx, :, :]

        return {
            "inputs": torch.from_numpy(np.array(x, dtype=np.float32, copy=True)),
            "labels": torch.from_numpy(np.array(y, dtype=np.float32, copy=True)),
        }


def build_dataloader(args, features, labels, indices=None, shuffle=True):
    dataset = BaselineDataset(
        features=features,
        labels=labels,
        indices=indices,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def load_model(args, input_size, output_size):
    cfg_path = os.path.join(os.path.dirname(__file__), "configs/timekimi_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config["input_size"] = input_size
    config["output_size"] = output_size
    config = KimiLinearConfig(**config)
    config.linear_attn_config["T_cycle"] = args.T_cycle
    
    model = KimiLinearTimeModel(config).to(DEVICE).to(MODEL_DTYPE)

    if torch.cuda.is_available() and not args.disable_data_parallel:
        if args.device_ids.strip():
            device_ids = [int(x.strip()) for x in args.device_ids.split(",") if x.strip()]
        else:
            device_ids = list(range(torch.cuda.device_count()))

        if len(device_ids) > 1:
            model = nn.DataParallel(model, device_ids=device_ids)
            print(f"Using DataParallel on GPUs: {device_ids}")

    model.train()
    
    def _count_params(model):
        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        total = sum(p.numel() for p in raw_model.parameters())
        trainable = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        return total, trainable
    
    total, trainable = _count_params(model)
    print(f"Total params: {total:,} ({total/1e6:.2f}M)")
    print(f"Trainable params: {trainable:,} ({trainable/1e6:.2f}M)")
    
    return model


def build_scheduler(args, optimizer, steps_per_epoch):
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(total_steps * args.warmup_ratio)
    warmup_steps = min(warmup_steps, max(total_steps - 1, 0))
    base_lr = args.learning_rate

    def lr_lambda(current_step):
        if total_steps <= 0:
            return 1.0
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)

        if total_steps == warmup_steps:
            return args.min_lr_ratio

        progress = (current_step - warmup_steps) / float(total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    if total_steps > 0:
        for param_group in optimizer.param_groups:
            param_group["lr"] = base_lr * lr_lambda(0)
    return scheduler, total_steps, warmup_steps


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


def evaluate_metrics(model, dataloader):
    model_was_training = model.training
    model.eval()

    with torch.no_grad():
        batch = next(iter(dataloader))
        inputs = batch["inputs"].to(DEVICE, dtype=MODEL_DTYPE)
        labels = batch["labels"]
        preds, _ = model(input_ids=inputs, use_cache=False)

    if model_was_training:
        model.train()

    preds = preds.detach().to(torch.float32).cpu().numpy()   # [B, L, C]
    labels = labels.to(torch.float32).cpu().numpy()          # [B, L, C]

    ic_values = []
    rank_ic_values = []
    seq_len = preds.shape[1]
    output_size = preds.shape[2]

    for t in range(seq_len):
        for c in range(output_size):
            pred_tc = preds[:, t, c]    # 所有股票，某一个评价因子的取值
            label_tc = labels[:, t, c]

            ic = _pearson_corr_1d(pred_tc, label_tc)
            if np.isfinite(ic):
                ic_values.append(ic)

            rank_ic = _pearson_corr_1d(_rankdata_1d(pred_tc), _rankdata_1d(label_tc))
            if np.isfinite(rank_ic):
                rank_ic_values.append(rank_ic)

    mean_ic = float(np.mean(ic_values)) if ic_values else float("nan")
    mean_rank_ic = float(np.mean(rank_ic_values)) if rank_ic_values else float("nan")

    if len(ic_values) >= 2:
        ic_std = float(np.std(ic_values, ddof=0))
        ir = float(mean_ic / ic_std) if ic_std > 0 else float("nan")
    else:
        ir = float("nan")

    return mean_ic, mean_rank_ic, ir


def plot_training_curves(output_dir, epoch_losses, epoch_ics, epoch_rank_ics, epoch_irs):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    axes[0, 0].plot(epoch_losses)
    axes[0, 0].set_title("Train Loss (MSE)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")

    axes[0, 1].plot(epoch_ics)
    axes[0, 1].set_title("Train IC")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("IC")

    axes[1, 0].plot(epoch_rank_ics)
    axes[1, 0].set_title("Train RankIC")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("RankIC")

    axes[1, 1].plot(epoch_irs)
    axes[1, 1].set_title("Train IR")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("IR")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "train_metrics_overview.png"), dpi=150)
    plt.close(fig)


def train(args, model, dataloader):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler, total_steps, warmup_steps = build_scheduler(args, optimizer, len(dataloader))
    
    every_log_steps = 10
    epoch_losses = []
    epoch_ics = []
    epoch_rank_ics = []
    epoch_irs = []
    global_step = 0
    print(f"total_steps={total_steps}, warmup_steps={warmup_steps}")
    
    for epoch in tqdm(range(args.epochs), desc="Epoch"):
        total_loss = 0.0

        for batch in dataloader:
            inputs = batch["inputs"].to(DEVICE, dtype=MODEL_DTYPE)
            labels = batch["labels"].to(DEVICE, dtype=MODEL_DTYPE)

            optimizer.zero_grad()
            
            preds, _ = model(
                input_ids=inputs,
                use_cache=False,
            )  # preds: [B, L, output_size]

            loss = criterion(preds.float(), labels.float())
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # 模型整体参数的梯度缩放
            
            optimizer.step()
            if global_step + 1 < total_steps:
                scheduler.step()

            total_loss += float(loss.detach().cpu())
            global_step += 1

            if global_step % every_log_steps == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"[epoch={epoch:03d} step={global_step:06d}] loss={float(loss.detach().cpu()):.6f} lr={current_lr:.6e}")

        avg_epoch_loss = total_loss / max(1, len(dataloader))
        epoch_losses.append(avg_epoch_loss)
        mean_ic, mean_rank_ic, ir = evaluate_metrics(model, dataloader)
        epoch_ics.append(mean_ic)
        epoch_rank_ics.append(mean_rank_ic)
        epoch_irs.append(ir)
        print(
            f"[epoch={epoch:03d}] avg_loss={avg_epoch_loss:.6f} "
            f"IC={mean_ic:.6f} RankIC={mean_rank_ic:.6f} IR={ir:.6f}"
        )

    plot_training_curves(
        output_dir=os.path.dirname(__file__),
        epoch_losses=epoch_losses,
        epoch_ics=epoch_ics,
        epoch_rank_ics=epoch_rank_ics,
        epoch_irs=epoch_irs,
    )

    # save weights
    save_dir = os.path.join(os.path.dirname(__file__), "CycleKDA_ckpt")
    os.makedirs(save_dir, exist_ok=True)
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(raw_model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))


def main():
    args = parse_args()
    
    print(f"device: {DEVICE}")
    
    features, labels, train_idx, test_idx = load_train_data(args)
    print(f"train samples: {len(train_idx)}, test samples: {len(test_idx)}")
    input_size = features.shape[-1]
    output_size = labels.shape[-1]
    dataloader = build_dataloader(args, features, labels, indices=train_idx)
    
    model = load_model(args, input_size, output_size)
    
    train(args, model, dataloader)


if __name__ == '__main__':
    main()
