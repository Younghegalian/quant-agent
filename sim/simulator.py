import numpy as np
import pandas as pd
from core.utils import load_config

cfg = load_config()
win15 = cfg["data"]["window_15m"]
win1d = cfg["data"]["window_1d"]

class Simulator:
    """
    실데이터 기반 거래 환경
    - df_15m: 15분봉 DataFrame (datetime index, OHLCV 등)
    - df_1d: 일봉 DataFrame (datetime index, OHLCV 등)
    - 환경의 시간 단위는 15분봉 기준으로 step 진행
    """
    def __init__(self, cfg, price_15m=None, price_1d=None):
        self.cfg = cfg
        self.fee = cfg["sim"]["fee"]
        self.init_krw = cfg["sim"]["init_krw"]

        # ===== ① 실데이터 로드 =====
        if price_15m is None or price_1d is None:
            raise ValueError("실데이터 DataFrame(price_15m, price_1d)을 입력해야 합니다.")

        # 인덱스 교집합 구간 계산
        start = max(price_15m.index[0], price_1d.index[0])
        end   = min(price_15m.index[-1], price_1d.index[-1])
        self.df_15m = price_15m.loc[start:end].copy()
        self.df_1d  = price_1d.loc[start:end].copy()

        # ===== ② 시뮬레이터 주요 변수 =====
        self.data_15m = self.df_15m["close"].to_numpy(dtype=np.float32)
        self.data_1d  = self.df_1d["close"].to_numpy(dtype=np.float32)
        self.kp_data  = np.full_like(self.data_15m, 2.0, dtype=np.float32)  # 김치프리미엄 더미

        # 루프 최대길이 = 15분봉 기준
        self.max_steps = len(self.data_15m)
        self.ptr = win15  # 윈도우 offset

        self.reset()

    # ===== ③ 환경 초기화 =====
    def reset(self):
        self.step_idx = 0
        self.krw = float(self.init_krw)
        self.usdt = 0.0
        self.value = self.krw
        self.ptr = win15
        return self._state()

    # ===== ④ 내부 헬퍼 =====
    def _price(self, i):
        i = max(0, min(i, len(self.data_15m) - 1))
        return float(self.data_15m[i])

    def _state(self):
        idx = self.step_idx

        w15 = self.data_15m[max(0, idx - win15):idx + 1]
        w1d = self.data_1d[max(0, idx - win1d):idx + 1]
        wkp = self.kp_data[max(0, idx - 8):idx + 1]

        return {
            "price_15m": w15.tolist(),
            "price_1d": w1d.tolist(),
            "kimchi_premium": wkp.tolist(),
            "krw_balance": self.krw,
            "usdt_balance": self.usdt,
            "current_price": self._price(idx),
        }

    # ===== ⑤ step =====
    def step(self, action_str):
        prev_value = self._portfolio_value()
        price = self._price(self.step_idx)

        # --- 거래 로직 ---
        if action_str == "BUY" and self.krw > 0:
            self.usdt = (self.krw / price) * (1 - self.fee)
            self.krw = 0.0
        elif action_str == "SELL" and self.usdt > 0:
            self.krw = (self.usdt * price) * (1 - self.fee)
            self.usdt = 0.0
        # HOLD는 아무것도 안함

        # --- 다음 스텝으로 ---
        self.step_idx += 1
        done = self.step_idx >= self.max_steps - 1

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

    # ===== ⑥ 포트폴리오 평가 =====
    def _portfolio_value(self):
        p = self._price(self.step_idx)
        return self.krw + self.usdt * p