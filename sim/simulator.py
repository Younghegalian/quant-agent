import numpy as np
from core.utils import load_config

cfg = load_config()
win15 = cfg["data"]["window_15m"]
win1d = cfg["data"]["window_1d"]

class Simulator:
    """
    실전 API 입력과 동일한 state 포맷을 흉내냄.
    - step(action_str) -> next_state, metrics, done
    metrics에는 보상계산용 value_prev/value_now 포함.
    """
    def __init__(self, cfg, price_15m=None):
        self.max_steps = cfg["sim"]["max_steps"]
        self.fee = cfg["sim"]["fee"]
        self.init_krw = cfg["sim"]["init_krw"]

        # 더미 시계열 (실제에선 과거 CSV close 사용)
        self.prices = price_15m if price_15m is not None else (
            np.cumsum(np.random.randn(self.max_steps).astype(np.float32)) + 1370
        )

        self.reset()

    def reset(self):
        self.step_idx = 0
        self.krw = float(self.init_krw)
        self.usdt = 0.0
        self.value = self.krw

        state = {
            "price_15m": self.data_15m[self.ptr - win15: self.ptr],
            "price_1d": self.data_1d[self.ptr - win1d: self.ptr],
            "kimchi_premium": self.kp_data[self.ptr - win1d: self.ptr],
            "krw_balance": self.krw,
            "usdt_balance": self.usdt,
            "current_price": self.data_15m[self.ptr],
        }

        return self._state()

    def _price(self, i):
        i = max(0, min(i, len(self.prices) - 1))
        return float(self.prices[i])

    def _state(self):
        idx = self.step_idx
        w = self.prices[max(0, idx-36):idx+1]  # 37개 윈도우
        return {
            "price_15m": w.tolist(),
            "price_1d": [float(np.mean(w))] * 8,
            "kimchi_premium": [2.0] * 8,
            "krw_balance": self.krw,
            "usdt_balance": self.usdt,
            "current_price": self._price(idx),
        }

    def step(self, action_str):
        prev_value = self._portfolio_value()
        price = self._price(self.step_idx)

        if action_str == "BUY" and self.krw > 0:
            self.usdt = (self.krw / price) * (1 - self.fee)
            self.krw = 0.0
        elif action_str == "SELL" and self.usdt > 0:
            self.krw = (self.usdt * price) * (1 - self.fee)
            self.usdt = 0.0
        # HOLD는 아무것도 안 함

        self.step_idx += 1
        done = self.step_idx >= len(self.prices) - 1
        next_state = self._state()

        now_value = self._portfolio_value()
        metrics = {
            "value_prev": prev_value,
            "value_now": now_value,
            "price": price,
            "krw_balance": self.krw,
            "usdt_balance": self.usdt,
        }
        return next_state, metrics, done

    def _portfolio_value(self):
        p = self._price(self.step_idx)  # 다음가 기준으로 평가해도 됨
        return self.krw + self.usdt * p