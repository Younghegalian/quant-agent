import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from .model import PolicyNetwork
from .ppo_core import PPOCore
from .memory import ReplayBuffer, Transition
from .utils import log, to_device

class RLAgent:
    def __init__(self, config: dict, mode: str = "sim"):
        """
        mode: 'sim' or 'live'
        """
        self.cfg = config
        self.mode = mode

        # 디바이스 설정
        self.device = torch.device(
            config["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        # 모델 초기화
        self.model = PolicyNetwork(**config["model"]).to(self.device)
        self.last_value = None
        self.save_dir = self.cfg.get("training", {}).get("save_dir", "checkpoints")
        os.makedirs(self.save_dir, exist_ok=True)

        # 학습 파라미터 선택 (mode에 따라 다르게)
        train_cfg = config["training"][mode]
        ppo_cfg = {}
        ppo_cfg.update({
            "lr": train_cfg["lr"],
            "clip_epsilon": train_cfg["clip_epsilon"],
            "entropy_coef": train_cfg["entropy_coef"],
            "value_coef": train_cfg["value_coef"],
            "update_epochs": train_cfg["update_epochs"],
        })

        # PPO Core 초기화
        self.ppo = PPOCore(self.model, **ppo_cfg)

        # 공통 파라미터
        self.gamma = config["ppo"]["gamma"]
        self.update_interval = train_cfg["update_interval"]
        self.save_dir = config["training"]["save_dir"]
        os.makedirs(self.save_dir, exist_ok=True)

        # 버퍼 및 상태 초기화
        self.buffer = ReplayBuffer(maxlen=config["training"]["buffer_maxlen"])
        self.global_steps = 0

        self.eval_mode = False
        log(f"[Agent] initialized on {self.device} | mode={mode}")

    # -------- inference ----------
    def act(self, state: dict) -> dict:
        """
        0=HOLD, 1=SIGNAL → SIGNAL 해석은 여기서 BUY/SELL로 변환
        + 매수는 현재가보다 1원 높게, 매도는 1원 낮게 주문하도록 조정
        HOLD일 경우 order_price=None 반환
        반환값: {'action': str, 'order_price': float or None}
        """
        # ✅ 세 개 다 받기
        x_short, x_long, x_acc = self._preprocess_state(state)

        with torch.no_grad():
            logits, value = self.model((x_short, x_long, x_acc))  # ✅ 튜플로 전달
            probs = F.softmax(logits, dim=-1)
            action_idx = (torch.argmax if self.eval_mode else torch.multinomial)(probs, 1).item()

        price = float(state.get("current_price"))
        krw = float(state.get("krw_balance"))
        usdt = float(state.get("usdt_balance"))
        total = krw + usdt * price
        ratio = 0.0 if total <= 0 else (usdt * price) / total
        th = self.cfg.get("policy", {}).get("signal_sell_threshold", 0.10)

        # ✅ 기본값 (HOLD)
        interpreted = "HOLD"
        order_price = None

        if action_idx == 1:
            if ratio >= th and usdt > 0:
                interpreted = "SELL"
                order_price = max(price - 1.0, 1200.0)  # ✅ 매도는 1원 낮게
            elif krw > 0:
                interpreted = "BUY"
                order_price = price + 1.0  # ✅ 매수는 1원 높게

        return {"action": interpreted, "order_price": order_price}

    # -------- reward ----------
    def compute_reward(self, state: dict, action=None) -> float:
        """
        강화학습용 트레이딩 리워드 함수 (개선 버전)
        -------------------------------------------------
        - Entry–Exit 실현수익률 기반 + 거래수수료 반영
        - HOLD 중 평가손익(unrealized PnL) 소폭 반영
        - 포지션 유지 시간 감쇠 및 반전 패널티 적용
        """

        # --- 액션 파싱 ---
        if isinstance(action, dict):
            action_str = action.get("action", "").upper()
            order_price = action.get("order_price")
        elif isinstance(action, (tuple, list)):
            action_str = str(action[0]).upper()
            order_price = action[1] if len(action) > 1 else None
        elif isinstance(action, str):
            action_str = action.upper()
            order_price = None
        else:
            action_str, order_price = None, None

        # --- 설정값 ---
        scale = self.cfg.get("policy", {}).get("reward_scale", 1.0)
        λ = self.cfg.get("policy", {}).get("hold_penalty_lambda", 0.001)
        fee = self.cfg.get("sim", {}).get("fee_rate", 0.0005)
        hp = self.cfg.get("policy", {}).get("hold_penalty", 0.02)
        pr = self.cfg.get("policy", {}).get("profit_scale", 0.1)

        # --- 내부 상태 초기화 ---
        if not hasattr(self, "position_open"):
            self.position_open = False
        if not hasattr(self, "entry_price"):
            self.entry_price = None
        if not hasattr(self, "hold_count"):
            self.hold_count = 0
        if not hasattr(self, "prev_action"):
            self.prev_action = None

        # --- 현재 시장가격 ---
        price = float(state.get("price", 0.0))
        avg_15m = np.mean(float(state.get("current_price")))
        reward = 0.0

        # =====================
        #  BUY (진입)
        # =====================
        if action_str == "BUY":
            self.position_open = True
            self.entry_price = order_price
            self.hold_count = 0
            low_price = (avg_15m - order_price) / max(avg_15m, 1e-8)
            reward = pr * torch.tanh(torch.tensor(low_price * 10.0)).item()

        # =====================
        #  SELL (청산)
        # =====================
        elif action_str == "SELL":
            exit_price = order_price

            # 수수료 반영 실현 손익률
            profit_ratio = ((exit_price * (1 - fee)) - (self.entry_price * (1 + fee))) / max(self.entry_price, 1e-8)

            # Tanh 스케일링 + global scale
            reward = pr * torch.tanh(torch.tensor(profit_ratio * 50.0)).item()

            # 포지션 종료
            self.position_open = False
            self.entry_price = None
            self.hold_count = 0

        # =====================
        #  HOLD (포지션 유지)
        # =====================
        elif action_str == "HOLD" and self.position_open:
            self.hold_count += 1

            # 평가손익 기반 부분 리워드 (unrealized)
            self.unrealized_ratio = (price - self.entry_price) / max(self.entry_price, 1e-8)
            reward = pr * torch.tanh(torch.tensor(self.unrealized_ratio * 5.0)).item()

            # 포지션 유지 감쇠 (시간 패널티)
            decay = min(λ * self.hold_count, hp)
            reward -= decay

        # =====================
        #  HOLD (비포지션 상태)
        # =====================
        elif action_str == "HOLD" and not self.position_open:
            self.hold_count += 1
            decay = min(λ * self.hold_count, hp)
            reward = -decay


        # =====================
        #  반전 패널티
        # =====================
        if (
                self.prev_action == "BUY" and action_str == "SELL"
        ) or (
                self.prev_action == "SELL" and action_str == "BUY"
        ):
            reward -= 0.08 # 포지션 반전 시 약한 패널티 (과도한 flip 방지)

        # --- 상태 업데이트 ---
        self.prev_action = action_str

        # --- Reward clipping ---
        reward = np.clip(reward, -scale, scale)

        return float(reward)

    # -------- experience ----------
    def store_transition(self, state, action, reward, next_state, done, aux=None):
        action_str = action.get("action")
        self.buffer.append(Transition(state, action_str, reward, next_state, done, aux))
        self.global_steps += 1

    def ready_to_learn(self) -> bool:
        return len(self.buffer) >= self.update_interval

    # -------- learning ----------
    def learn(self):
        if not self.ready_to_learn():
            return None

        batch = self.buffer.sample(self.update_interval)
        states_s, states_l, states_acc = [], [], []
        actions_idx, rewards, dones = [], [], []

        # 1) 배치 전처리 (한 번에 스택 → 한 번에 forward)
        with torch.no_grad():
            for tr in batch:
                xs, xl, acc = self._preprocess_state(tr.state)  # ✅ 세 개 다 언팩
                states_s.append(xs.squeeze(0))
                states_l.append(xl.squeeze(0))
                states_acc.append(acc.squeeze(0))
                actions_idx.append(0 if tr.action == "HOLD" else 1)
                rewards.append([tr.reward])
                dones.append([1.0 if tr.done else 0.0])

        # 2) 텐서 스택
        states_s = torch.stack(states_s).to(self.device)  # (B, Ts, Fs)
        states_l = torch.stack(states_l).to(self.device)  # (B, Tl, Fl)
        states_acc = torch.stack(states_acc).to(self.device)  # (B, 3)
        actions = torch.tensor(actions_idx, dtype=torch.long, device=self.device).unsqueeze(-1)  # (B,1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)  # (B,1)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)  # (B,1)

        # 3) old logits/values (첫 pass)
        with torch.no_grad():
            old_logits, values = self.model((states_s, states_l, states_acc))

        # 4) returns & advantages 계산
        returns = []
        G = torch.zeros(1, device=self.device)
        for r, d in zip(reversed(rewards), reversed(dones)):
            G = r + self.gamma * G * (1 - d)
            returns.insert(0, G)
        returns = torch.stack(returns).squeeze(1).detach()  # (B,)
        advantages = (returns - values.squeeze(1)).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 5) new forward
        logits, new_values = self.model((states_s, states_l, states_acc))  # ✅ acc 추가 반영

        # 6) PPO loss 계산 및 업데이트
        loss = self.ppo.compute_loss(
            logits=logits,
            old_logits=old_logits,
            actions=actions,
            advantages=advantages.unsqueeze(-1),  # (B,1)
            values=new_values,
            returns=returns.unsqueeze(-1)  # (B,1)
        )

        self.ppo.step(loss)
        self.buffer.clear()

        return float(loss.item())

        # -------- utils ----------

    def _preprocess_state(self, state: dict):

        # --- Short: price_15m ---
        s = np.array(state.get("price_15m", []), dtype=np.float32)
        if s.ndim == 1: s = s[:, None]
        if s.size > 0:
            s = (s - s.mean(axis=0, keepdims=True)) / (s.std(axis=0, keepdims=True) + 1e-8)
        xs = torch.tensor(s, dtype=torch.float32).unsqueeze(0)

        # --- Long: price_1d + kimchi_premium ---
        p1d = np.array(state.get("price_1d", []), dtype=np.float32)
        kp = np.array(state.get("kimchi_premium", []), dtype=np.float32)
        L = min(len(p1d), len(kp))
        xl_np = np.column_stack([p1d[:L], kp[:L]]) if L > 0 else np.zeros((0, 2), np.float32)
        if xl_np.size > 0:
            xl_np = (xl_np - xl_np.mean(axis=0, keepdims=True)) / (xl_np.std(axis=0, keepdims=True) + 1e-8)
        xl = torch.tensor(xl_np, dtype=torch.float32).unsqueeze(0)


        # --- Account: ratio + current_price ---
        price = float(state.get("current_price", 0.0))

        if hasattr(self, "unrealized_ratio") and self.unrealized_ratio:
            pnl_ratio = self.unrealized_ratio
        else:
            pnl_ratio = 0.0

        krw = float(state.get("krw_balance", 0.0))
        usdt = float(state.get("usdt_balance", 0.0))
        total = krw + usdt * price
        ratio = 0.0 if total <= 0 else (usdt * price) / total
        acc_vec = np.array([ratio, price, pnl_ratio], dtype=np.float32)[None, :]  # (1,3)
        acc = torch.tensor(acc_vec, dtype=torch.float32)



        return to_device(xs, self.device), to_device(xl, self.device), to_device(acc, self.device)

    def save(self, path: str = None):
        """
        모델과 PPO optimizer 상태를 저장.
        path를 생략하면 config에 맞춰 자동 경로로 저장.
        """
        if path is None:
            prefix = self.cfg.get("live", {}).get("save_prefix", "live_model")
            ts = int(time.time())
            path = os.path.join(self.save_dir, f"{prefix}_{ts}.pth")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": (
                    self.ppo.optimizer.state_dict() if hasattr(self, "ppo") else None
                ),
                "config": self.cfg,
            },
            path,
        )
        return path  # 저장 경로 반환

    def load(self, path: str, strict: bool = True):
        """저장된 체크포인트를 복원."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=strict)

        opt_state = ckpt.get("optimizer_state_dict")
        if opt_state and hasattr(self, "ppo") and hasattr(self.ppo, "optimizer"):
            self.ppo.optimizer.load_state_dict(opt_state)

        # 저장 당시 config 반영(선택)
        self.cfg = ckpt.get("config", self.cfg)
        # 저장 디렉토리 동기화
        self.save_dir = self.cfg.get("training", {}).get("save_dir", self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        return path