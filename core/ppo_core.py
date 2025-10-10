import torch
import torch.nn.functional as F
from torch import optim


class PPOCore:
    def __init__(self, model, lr, clip_epsilon, entropy_coef, value_coef, update_epochs, max_grad_norm=1.0):
        self.model = model
        self.lr = lr
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.update_epochs = update_epochs
        self.max_grad_norm = max_grad_norm
        self.opt = optim.Adam(self.model.parameters(), lr=self.lr)
        self.optimizer = self.opt

    # ------------------------------------------------------------
    def compute_loss(self, logits, old_logits, actions, advantages, values, returns):
        """
        PPO 손실 계산
        logits, old_logits: (B, num_actions)
        actions: (B,1)
        advantages, values, returns: (B,1)
        """
        logits = torch.clamp(logits, -20, 20)
        old_logits = torch.clamp(old_logits, -20, 20)

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
        probs_new = F.softmax(logits, dim=-1)
        probs_old = F.softmax(old_logits, dim=-1)
        kl = (probs_old * (torch.log(probs_old + 1e-8) - torch.log(probs_new + 1e-8))).sum(dim=1).mean()
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy + 0.5 * kl

        return loss

    # ------------------------------------------------------------
    def step(self, loss):
        """PPO step"""
        self.opt.zero_grad(set_to_none=True)
        loss.backward()

        if self.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)

        self.opt.step()