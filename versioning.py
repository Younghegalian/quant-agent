import os
import yaml
import torch
from datetime import datetime

class ExperimentVersion:
    def __init__(self, base_dir="checkpoints", config=None):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name = f"exp_{self.timestamp}"
        self.base_dir = base_dir
        self.exp_dir = os.path.join(base_dir, self.exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)

        # config.yaml 저장
        if config is not None:
            with open(os.path.join(self.exp_dir, "config.yaml"), "w") as f:
                yaml.dump(config, f)

        print(f"[versioning] New experiment created: {self.exp_name}")

    def save_checkpoint(self, model, step):
        path = os.path.join(self.exp_dir, f"model_step_{step}.pt")
        model_state = model.state_dict()
        torch.save(model_state, path)
        print(f"[versioning] Saved checkpoint: {path}")
        return path