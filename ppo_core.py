import torch
import torch.nn.functional as F

class PPOCore:
    def __init__(self, model, lr=3e-4, clip_eps=0.2, gamma=0.99):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.clip_eps = clip_eps
        self.gamma = gamma

    def compute_loss(self, logits, old_logits, actions, advantages, values, returns):
        probs = F.softmax(logits, dim=-1)
        old_probs = F.softmax(old_logits, dim=-1).detach()

        # ✅ actions shape 정리 (무조건 2D로)
        if actions.ndim == 1:
            actions = actions.unsqueeze(1)
        elif actions.ndim > 2:
            actions = actions.view(-1, 1)

        # ✅ advantages, returns 차원 맞춤
        if advantages.ndim == 1:
            advantages = advantages.unsqueeze(1)
        if returns.ndim == 1:
            returns = returns.unsqueeze(1)
        if values.ndim == 1:
            values = values.unsqueeze(1)

        # ratio 계산
        act_prob = probs.gather(1, actions)
        old_act_prob = old_probs.gather(1, actions)
        ratio = act_prob / (old_act_prob + 1e-8)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values, returns)

        return policy_loss + 0.5 * value_loss

    def step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()