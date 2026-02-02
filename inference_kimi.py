import os
import json
import math
import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from KimiLinear import KimiLinearConfig, KimiLinearTimeModel


class CosineDataset(Dataset):
    def __init__(self, num_samples: int = 10000, seq_len: int = 128):
        super().__init__()
        self.seq_len = seq_len

        step = math.pi / 8
        total_points = num_samples + seq_len + 1
        t = torch.arange(total_points, dtype=torch.float32) * step
        data = torch.cos(t) * 4

        xs, ys = [], []
        for i in range(num_samples):
            x = data[i : i + seq_len]
            y = data[i + 1 : i + seq_len + 1]
            xs.append(x)
            ys.append(y)

        self.x = torch.stack(xs).unsqueeze(-1)  # [N, L, 1]
        self.y = torch.stack(ys).unsqueeze(-1)  # [N, L, 1]

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return {"inputs": self.x[idx], "labels": self.y[idx]}


@torch.no_grad()
def autoregressive_rollout(
    model: nn.Module,
    context: torch.Tensor,         # [B, C, 1]
    rollout_len: int,
    use_cache: bool = True,
):
    """
    返回 preds: [B, rollout_len, 1]
    """
    model.eval()

    past = None

    # 1) 用 context 建 cache，并拿到“预测下一个点”的输出（取最后一个位置）
    out, past = model(
        input_ids=context,
        past_key_values=past,
        use_cache=use_cache,
    )  # out: [B, C, 1]
    next_token = out[:, -1:, :]  # [B, 1, 1]

    preds = [next_token]

    # 2) 纯自回归：后续每一步只喂上一步预测
    for _ in range(1, rollout_len):
        out_step, past = model(
            input_ids=next_token,
            past_key_values=past,
            use_cache=use_cache,
        )  # out_step: [B, 1, 1]
        next_token = out_step[:, -1:, :]
        preds.append(next_token)

    preds = torch.cat(preds, dim=1)  # [B, rollout_len, 1]
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="timekimi_ckpt")
    parser.add_argument("--num_samples", type=int, default=1024)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--context_len", type=int, default=32)
    parser.add_argument("--rollout_len", type=int, default=64)
    parser.add_argument("--out_png", type=str, default="inference_rollout.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # 载入 config + 权重
    cfg_path = os.path.join(args.ckpt_dir, "config.json")
    weight_path = os.path.join(args.ckpt_dir, "pytorch_model.bin")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"找不到 {cfg_path}")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"找不到 {weight_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg_json = json.load(f)

    config = KimiLinearConfig(**cfg_json)
    model = KimiLinearTimeModel(config).to(device).to(dtype)
    sd = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval()

    # 数据
    ds = CosineDataset(num_samples=args.num_samples, seq_len=args.seq_len)
    sample = ds[args.sample_idx]
    x = sample["inputs"].to(torch.bfloat16).unsqueeze(0).to(device=device, dtype=dtype)   # [1, L, 1]
    y = sample["labels"].to(torch.bfloat16).unsqueeze(0).to(device=device, dtype=dtype)   # [1, L, 1]

    assert args.context_len >= 1
    assert args.context_len <= args.seq_len
    # 目标对齐：context 的最后一个位置预测的是 labels 的 index = context_len-1
    max_roll = args.seq_len - (args.context_len - 1)
    rollout_len = min(args.rollout_len, max_roll)

    context = x[:, : args.context_len, :]  # [1, C, 1]

    # 自回归 rollout（用 cache）
    if device.type == "cuda":
        preds = autoregressive_rollout(model, context, rollout_len=rollout_len, use_cache=True)
    else:
        preds = autoregressive_rollout(model, context, rollout_len=rollout_len, use_cache=True)

    # GT：labels 从 (context_len-1) 开始往后 rollout_len 个
    gt = y[:, (args.context_len - 1) : (args.context_len - 1 + rollout_len), :]  # [1, rollout_len, 1]

    # 计算 MSE（仅用于观测）
    mse = torch.mean((preds.float() - gt.float()) ** 2).item()
    print(f"device={device}, dtype={dtype}, rollout_len={rollout_len}, mse={mse:.6f}")

    # 画图（转 float32）
    gt_np = gt[0, :, 0].float().cpu().numpy()
    pd_np = preds[0, :, 0].float().cpu().numpy()

    plt.figure(figsize=(8, 3))
    plt.plot(gt_np, label="gt (x[t+1])")
    plt.plot(pd_np, label="pred (AR rollout)")
    plt.title(f"AR rollout mse={mse:.6f}")
    plt.legend()
    plt.tight_layout()

    out_path = args.out_png
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()