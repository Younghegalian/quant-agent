import os
import torch
import torch.nn.functional as F
from .model import PolicyNetwork
from .ppo_core import PPOCore
from .memory import ReplayBuffer, Transition
from .utils import log, to_device


class RLAgent:
    def __init__(self, config: dict):
        self.cfg = config
        self.device = torch.device(
            config["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = PolicyNetwork(**config["model"]).to(self.device)
        self.ppo = PPOCore(self.model, **config["ppo"])

        self.gamma = config["ppo"]["gamma"]
        self.update_interval = config["training"]["update_interval"]
        self.save_dir = config["training"]["save_dir"]
        os.makedirs(self.save_dir, exist_ok=True)

        self.buffer = ReplayBuffer(maxlen=config["training"]["buffer_maxlen"])
        self.global_steps = 0
        log(f"[Agent] initialized on {self.device}")

    # -------- inference ----------
    def act(self, state: dict) -> str:
        """0=HOLD, 1=SIGNAL → SIGNAL 해석은 여기서 BUY/SELL로 변환"""
        x = self._preprocess_state(state)  # (1,T,F)
        with torch.no_grad():
            logits, value = self.model(x)
            probs = F.softmax(logits, dim=-1)  # (1,2)
            action_idx = torch.multinomial(probs, num_samples=1).item()  # 0 or 1

        interpreted = "HOLD"
        if action_idx == 1:
            # 계좌 상태 기반 SIGNAL 해석
            price = state.get("current_price", 1.0)
            krw = float(state.get("krw_balance", 0.0))
            usdt = float(state.get("usdt_balance", 0.0))
            total = krw + usdt * price
            ratio = 0.0 if total <= 0 else (usdt * price) / total
            th = self.cfg.get("policy", {}).get("signal_sell_threshold", 0.10)
            interpreted = "SELL" if ratio >= th and usdt > 0 else ("BUY" if krw > 0 else "HOLD")

        return interpreted

    # -------- reward ----------
    def compute_reward(self, metrics: dict) -> float:
        """
        실전/시뮬 공통: env/executor는 raw metrics만 제공.
        여기서 보상 계산을 정의한다.
        기본: 자산 변화율을 tanh 스케일링.
        """
        prev_v = metrics.get("value_prev", None)
        now_v  = metrics.get("value_now", None)
        if prev_v is None or now_v is None or prev_v <= 0:
            return 0.0
        delta = (now_v - prev_v) / prev_v
        # 안정화
        return float(torch.tanh(torch.tensor(delta * 5.0)).item())

    # -------- experience ----------
    def store_transition(self, state, action_str, reward, next_state, done, aux=None):
        self.buffer.append(Transition(state, action_str, reward, next_state, done, aux))
        self.global_steps += 1

    def ready_to_learn(self) -> bool:
        return len(self.buffer) >= self.update_interval

    # -------- learning ----------
    def learn(self):
        if not self.ready_to_learn():
            return None

        batch = self.buffer.sample(self.update_interval)
        # (간소화) 행동을 index로 매핑: HOLD=0, BUY=1, SELL=1 (signal은 어차피 1로 학습)
        # → 정책은 "signal 낼 타이밍"만 학습, 해석은 act()에서 계좌 상태로 분기.
        states, actions_idx, rewards, dones, old_logits, values = [], [], [], [], [], []

        with torch.no_grad():
            for tr in batch:
                x = self._preprocess_state(tr.state)  # (1,T,F)
                logits, val = self.model(x)
                old_logits.append(logits.squeeze(0))  # (2,)
                values.append(val.squeeze(0))         # (1,)
                actions_idx.append(0 if tr.action == "HOLD" else 1)
                rewards.append([tr.reward])
                dones.append([1.0 if tr.done else 0.0])
                states.append(x.squeeze(0))

        states  = torch.stack(states).to(self.device)            # (B,T,F)
        actions = torch.tensor(actions_idx, dtype=torch.long, device=self.device).unsqueeze(-1)  # (B,1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)                 # (B,1)
        dones   = torch.tensor(dones, dtype=torch.float32, device=self.device)                   # (B,1)
        old_logits = torch.stack(old_logits).to(self.device)      # (B,2)
        values     = torch.stack(values).to(self.device)          # (B,1)

        # returns & advantages
        returns = []
        G = torch.zeros(1, device=self.device)
        for r, d in zip(reversed(rewards), reversed(dones)):
            G = r + self.gamma * G * (1 - d)
            returns.insert(0, G)
        returns = torch.stack(returns).squeeze(1).detach()  # (B,1) -> (B,)
        advantages = (returns - values.squeeze(1)).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # new forward
        logits, new_values = self.model(states)  # (B,2), (B,1)

        loss = self.ppo.compute_loss(
            logits=logits,
            old_logits=old_logits,
            actions=actions,
            advantages=advantages.unsqueeze(-1),  # (B,1)
            values=new_values,
            returns=returns.unsqueeze(-1)         # (B,1)
        )
        self.ppo.step(loss)
        return float(loss.item())

    # -------- io ----------
    # TODO: 파일 IO 에러핸들링 처리 필요
    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
        log(f"[Agent] saved: {path}")

    def load(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        log(f"[Agent] loaded: {path}")

    # -------- utils ----------
    def _preprocess_state(self, state: dict) -> torch.Tensor:
        """
        최소 전처리: (1,T,F)로 맞춘다. 여기선 예시로 close만 1채널로 사용.
        실전에선 adapter에서 state dict을 바로 (T,F)로 만들어 주면 됨.
        """
        import numpy as np
        # 예: state["price_15m"] 시퀀스 사용
        seq = state.get("price_15m", [])
        arr = np.array(seq, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]  # (T,1)
        # 표준화(옵션)
        if arr.size > 1:
            arr = (arr - arr.mean()) / (arr.std() + 1e-8)
        x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # (1,T,F)
        return to_device(x, self.device)