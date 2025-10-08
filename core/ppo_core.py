import torch
import torch.nn.functional as F
from torch import optim


class PPOCore:
    def __init__(self, model, lr=3e-4, clip_epsilon=0.2, entropy_coef=0.01, value_coef=0.5, update_epochs=3, **_):
        self.model = model
        self.opt = optim.Adam(model.parameters(), lr=lr)
        self.clip = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.update_epochs = update_epochs

    def compute_loss(self, logits, old_logits, actions, advantages, values, returns):
        # logits/old_logits: (B,2), actions: (B,1)
        new_logp = F.log_softmax(logits, dim=-1)
        old_logp = F.log_softmax(old_logits, dim=-1).detach()

        act_new_logp = new_logp.gather(1, actions)         # (B,1)
        act_old_logp = old_logp.gather(1, actions)         # (B,1)

        ratio = torch.exp(act_new_logp - act_old_logp)     # (B,1)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(values, returns)

        entropy = -(new_logp * torch.exp(new_logp)).sum(dim=1).mean()
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        return loss

    def step(self, loss):
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.opt.step()