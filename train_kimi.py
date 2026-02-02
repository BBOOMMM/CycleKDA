import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
import os
import sys
import math
from KimiLinear import KimiLinearConfig, KimiLinearTimeModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn

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

class CosineDataset(Dataset):
    def __init__(self, num_samples: int = 10000, seq_len: int = 128):
        # super().__init__()
        # self.seq_len = seq_len

        # t = torch.linspace(0, 2*math.pi, steps=10000)
        # data = torch.cos(t)  # [T_total]
        
        # # 生成一长段 cos 序列
        # # t = torch.linspace(0, 25, steps=num_samples + seq_len + 1)
        # # data = torch.cos(t)  # [T_total]

        # xs, ys = [], []
        # for i in range(num_samples):
        #     x = data[i:i + seq_len]          # 输入: 长度 L
        #     y = data[i + 1:i + seq_len + 1]  # 标签: 向右平移 1
        #     xs.append(x)
        #     ys.append(y)
        # self.x = torch.stack(xs).unsqueeze(-1)  # [N, L, 1]
        # self.y = torch.stack(ys).unsqueeze(-1)  # [N, L, 1]
        
        super().__init__()
        self.seq_len = seq_len

        # 以 π/8 为步长：0, π/8, 2π/8, 3π/8, ...
        step = math.pi / 8
        # 需要的总点数 = 可取样的起点个数 num_samples + 每段长度 seq_len + 1（因为要右移一位做 label）
        total_points = num_samples + seq_len + 1
        t = torch.arange(total_points, dtype=torch.float32) * step   # [total_points]
        data = torch.cos(t)*4                                          # [total_points]

        xs, ys = [], []
        for i in range(num_samples):
            x = data[i : i + seq_len]          # 输入: 长度 L
            y = data[i + 1 : i + seq_len + 1]  # 标签: 向右平移 1
            xs.append(x)
            ys.append(y)
        self.x = torch.stack(xs).unsqueeze(-1).to(torch.bfloat16)  # [N, L, 1]
        self.y = torch.stack(ys).unsqueeze(-1).to(torch.bfloat16)  # [N, L, 1]

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return {
            "inputs": self.x[idx],   # [L, 1]
            "labels": self.y[idx],   # [L, 1]
        }


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    num_samples = 1024
    seq_len = 128
    batch_size = 64
    num_epochs = 50
    lr = 1e-4

    dataset = CosineDataset(num_samples=num_samples, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    cfg_path = os.path.join(os.path.dirname(__file__), "configs/timekimi_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config["input_size"] = 1    # 输入维度：单变量时间序列

    config = KimiLinearConfig(**config)
    model = KimiLinearTimeModel(config).to(device).to(torch.bfloat16)
    model.train()

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    every_log_steps = 10
    step = 0
    epoch_losses = []
    global_step = 0
    
    for epoch in tqdm(range(num_epochs), desc="Epoch"):
        total_loss = 0.0

        for batch in dataloader:
            inputs = batch["inputs"].to(device)   # [B, L, 1]  作为 x[t]
            labels = batch["labels"].to(device)   # [B, L, 1]  作为 x[t+1]

            optimizer.zero_grad(set_to_none=True)

            # 训练期不使用 cache：并行更快，显存更省
            preds, _ = model(
                input_ids=inputs,   # 对 TimeModel 来说是连续值序列
                use_cache=False,
            )  # preds: [B, L, input_size]

            loss = criterion(preds, labels)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            global_step += 1

            if global_step % every_log_steps == 0:
                print(f"[epoch={epoch:03d} step={global_step:06d}] loss={float(loss.detach().cpu()):.6f}")

        avg_epoch_loss = total_loss / max(1, len(dataloader))
        epoch_losses.append(avg_epoch_loss)

        # 可选：每 20 个 epoch 画一次预测 vs label（全并行输出）
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                batch0 = next(iter(dataloader))
                x0 = batch0["inputs"].to(device)
                y0 = batch0["labels"].to(device)
                p0, _ = model(input_ids=x0, use_cache=False)

                i = 0
                gt = y0[i, :, 0].to(torch.float32).detach().cpu().numpy()
                pd = p0[i, :, 0].to(torch.float32).detach().cpu().numpy()

                plt.figure(figsize=(7, 3))
                plt.plot(gt, label="gt (x[t+1])")
                plt.plot(pd, label="pred")
                plt.title(f"epoch={epoch+1}, mse={avg_epoch_loss:.6f}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(os.path.dirname(__file__), "train_pred_debug.png"), dpi=150)
                plt.close()
            model.train()

    # loss 曲线
    plt.figure(figsize=(6, 3))
    plt.plot(epoch_losses)
    plt.title("train loss (MSE)")
    plt.xlabel("epoch")
    plt.ylabel("mse")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "train_loss.png"), dpi=150)
    plt.close()

    # 保存权重
    save_dir = os.path.join(os.path.dirname(__file__), "timekimi_ckpt")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, ensure_ascii=False, indent=2)

            



if __name__ == "__main__":
    train()











# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# cfg_path = os.path.join(os.path.dirname(__file__), "kimi_config.json")
# with open(cfg_path, "r", encoding="utf-8") as f:
#     cfg_json = json.load(f)

# config = KimiLinearConfig(**cfg_json)

# attn = KimiDeltaAttention(config, 0).to(device)
# attn.train()  # 开启训练模式

# batch_size = 2
# seq_len = 128
# hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device=device, dtype=torch.float32)

# attention_mask = None

# cache = KimiDynamicCache(config)

# optimizer = torch.optim.AdamW(attn.parameters(), lr=1e-4)

# out = attn(hidden_states, attention_mask=attention_mask, cache_params=cache)

# if isinstance(out, tuple):
#     o = out[0]
# else:
#     o = out

# loss = o.float().pow(2).mean()

# optimizer.zero_grad(set_to_none=True)
# loss.backward()

# torch.nn.utils.clip_grad_norm_(attn.parameters(), max_norm=1.0)

# optimizer.step()

# print("loss =", float(loss.detach().cpu()))
# print("o.shape =", tuple(o.shape))

