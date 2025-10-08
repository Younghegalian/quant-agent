import torch
import torch.nn.functional as F
from model import PolicyNetwork
from state_schema import preprocess_state
from utils import log
from ppo_core import PPOCore
import os
from datetime import datetime
from versioning import ExperimentVersion

class RLAgent:
    def __init__(self, config=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PolicyNetwork().to(self.device)
        self.ppo = PPOCore(self.model, lr=3e-4)
        self.buffer = []
        self.version = ExperimentVersion(config=config)
        log(f"RLAgent initialized on {self.device}")

    # ------------------------------------------------------------
    def act(self, state):
        """현재 state에서 행동 샘플링"""
        state_tensor = preprocess_state(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.model(state_tensor)

            # ✅ logits shape 방탄 처리 (1, 2) → (2)
            if logits.ndim > 1:
                logits = logits.squeeze(0)

            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1)

        # ✅ detach + float precision 고정
        return action, logits.detach().float(), value.detach().float()

    # ------------------------------------------------------------
    def store_transition(self, transition):
        """(state, action, reward, next_state, done, logits, value)"""
        self.buffer.append(transition)

    # ------------------------------------------------------------
    def learn(self):
        """버퍼에 쌓인 transition으로 PPO 업데이트"""
        if len(self.buffer) < 5:
            return  # 아직 학습할 만큼 안 쌓임

        log(f"Learning step with {len(self.buffer)} samples")

        states, actions, rewards, next_states, dones, old_logits, values = zip(*self.buffer)
        self.buffer.clear()

        states = torch.stack([preprocess_state(s) for s in states]).to(self.device)
        actions = torch.cat(actions, dim=0).long().to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1).to(self.device)

        old_logits = torch.stack(old_logits)
        if old_logits.ndim == 3:
            old_logits = old_logits.squeeze(1)
        old_logits = old_logits.to(self.device)

        values = torch.stack(values).to(self.device)

        # Advantage 계산
        returns = []
        G = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            G = r + 0.99 * G * (1 - d)
            returns.insert(0, G)
        returns = torch.stack(returns).detach()
        advantages = returns - values.detach()

        logits, new_values = self.model(states)
        if logits.ndim == 3:
            logits = logits.squeeze(1)

        loss = self.ppo.compute_loss(
            logits, old_logits, actions, advantages, new_values, returns
        )

        self.ppo.step(loss)
        log(f"Loss: {loss.item():.6f}")

        # ✅------------------- Checkpoint 저장 -------------------✅
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_dir = os.path.join("checkpoints", timestamp)
        os.makedirs(ckpt_dir, exist_ok=True)

        ckpt_path = os.path.join(ckpt_dir, f"model_step_{len(self.buffer)}.pt")
        torch.save(self.model.state_dict(), ckpt_path)
        log(f"Checkpoint saved: {ckpt_path}")

        self.version.save_checkpoint(self.model, step=len(self.buffer))