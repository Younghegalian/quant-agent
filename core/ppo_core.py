import torch
import torch.nn.functional as F
from torch import optim


class PPOCore:
    def __init__(self, model, cfg):
        """
        PPO core module.
        """
        self.model = model
        ppo_cfg = cfg.get("ppo", {})

        # Hyperparameters from config
        self.lr            = float(ppo_cfg.get("lr", 3e-4))
        self.clip_epsilon  = float(ppo_cfg.get("clip_ratio", 0.2))
        self.entropy_coef  = float(ppo_cfg.get("entropy_coef", 0.01))
        self.value_coef    = float(ppo_cfg.get("value_coef", 0.5))
        self.update_epochs = int(ppo_cfg.get("update_epochs", 3))
        self.max_grad_norm = float(ppo_cfg.get("max_grad_norm", 1.0))

        # Optimizer
        self.opt = optim.Adam(self.model.parameters(), lr=self.lr)

    # ------------------------------------------------------------
    def compute_loss(self, logits, old_logits, actions, advantages, values, returns):
        """
        PPO 손실 계산
        logits, old_logits: (B, num_actions)
        actions: (B,1)
        advantages, values, returns: (B,1)
        """
        # Log prob 계산
        new_logp = F.log_softmax(logits, dim=-1)
        old_logp = F.log_softmax(old_logits, dim=-1).detach()

        act_new_logp = new_logp.gather(1, actions)
        act_old_logp = old_logp.gather(1, actions)

        # ratio 계산 (log 확률 차이 → exp)
        ratio = torch.exp(act_new_logp - act_old_logp)

        # PPO objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # value loss (critic)
        value_loss = F.mse_loss(values, returns)

        # entropy (policy entropy는 양수여야 함)
        probs = torch.exp(new_logp)
        entropy = -(probs * new_logp).sum(dim=1).mean()

        # total loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        return loss

    # ------------------------------------------------------------
    def step(self, loss):
        """PPO step"""
        self.opt.zero_grad(set_to_none=True)
        loss.backward()

        if self.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)

        self.opt.step()