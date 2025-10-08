import torch
import numpy as np
from datetime import datetime


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def timestamp():
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")


def log(msg):
    print(f"{timestamp()} {msg}")