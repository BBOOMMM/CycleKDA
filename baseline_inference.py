import os
import json
import math
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from KimiLinear import KimiLinearConfig, KimiLinearTimeModel


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
    parser = argparse.ArgumentParser(description="CycleKDA baseline")
    parser.add_argument("--ckpt_dir", type=str, default="timekimi_ckpt")
    parser.add_argument("--num_samples", type=int, default=1024)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--sample_idx", type=int, default=1)
    parser.add_argument("--context_len", type=int, default=32)
    parser.add_argument("--rollout_len", type=int, default=64)
    parser.add_argument("--out_png", type=str, default="inference_rollout.png")
    args = parser.parse_args()