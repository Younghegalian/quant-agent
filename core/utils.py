import time, torch
import yaml
import os

def load_config(path=None):

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = path or os.path.join(base_dir, "config.yaml")

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def log(msg: str):
    print(msg, flush=True)

def to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return x

def ts():
    return time.strftime("%Y%m%d_%H%M%S")

def save_path(base_dir, name):
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"{name}")