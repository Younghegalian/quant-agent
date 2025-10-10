import numpy as np
import pandas as pd
from core.utils import load_config

cfg = load_config()
win15 = cfg["data"]["window_15m"]
win1d = cfg["data"]["window_1d"]

class Simulator:
    """
    실데이터 기반 거래 환경 (안정형)
    - 15분봉 단위 step 진행
    - 일봉 윈도우가 완전히 쌓인 시점부터 시작
    """

    def __init__(self, cfg, price_15m=None, price_1d=None):
        self.cfg = cfg
        self.fee = cfg["sim"]["fee"]
        self.init_krw = cfg["sim"]["init_krw"]

        if price_15m is None or price_1d is None:
            raise ValueError("실데이터 DataFrame(price_15m, price_1d)을 입력해야 합니다.")

        # ===== 교집합 구간 =====
        start = max(price_15m.index[0], price_1d.index[0])
        end   = min(price_15m.index[-1], price_1d.index[-1])
        price_15m = price_15m.loc[start:end].copy()
        price_1d  = price_1d.loc[start:end].copy()

        # ===== 데이터 구성 =====
        self.df_15m = price_15m
        self.df_1d = price_1d
        self.data_15m = price_15m["close"].to_numpy(dtype=np.float32)
        self.data_1d  = price_1d["close"].to_numpy(dtype=np.float32)
        self.kp_data  = price_1d["kp"].to_numpy(dtype=np.float32)

        self.max_steps = len(self.data_15m)

        # ===== 시작 지점 계산 =====
        if len(price_1d.index) <= win1d:
            raise ValueError("일봉 데이터 길이가 window보다 짧습니다.")

        # 일봉 윈도우가 완성된 시점
        start_ts = price_1d.index[win1d]
        # 15분봉에서 해당 시점 이상인 인덱스 찾기
        start_offset = np.searchsorted(price_15m.index, start_ts)
        # 15분봉 윈도우도 고려해서 보정
        self.start_offset = max(start_offset, win15)

        # 15분봉 길이 초과 방지
        if self.start_offset >= len(self.data_15m):
            self.start_offset = len(self.data_15m) - 1

        print(f"[Simulator] len(15m)={len(self.data_15m)}, len(1d)={len(self.data_1d)}, start_offset={self.start_offset}")
        self.reset()

    # ===== 환경 초기화 =====
    def reset(self):
        self.step_idx = self.start_offset
        self.krw = float(self.init_krw)
        self.usdt = 0.0
        self.value = self.krw
        self.ptr = self.step_idx
        return self._state()

    # ===== 내부 헬퍼 =====
    def _price(self, i):
        i = max(0, min(i, len(self.data_15m) - 1))
        return float(self.data_15m[i])

    def _state(self):
        idx = self.step_idx

        w15 = self.data_15m[max(0, idx - win15):idx]
        w1d = self.data_1d[max(0, idx - win1d):idx]
        wkp = self.kp_data[max(0, idx - win1d):idx]

        # === 길이 보정 (pad) ===
        def pad(arr, target):
            if len(arr) == 0:
                # 완전히 비어 있으면 0으로 채움
                return np.zeros(target, dtype=np.float32)
            if len(arr) < target:
                pad_len = target - len(arr)
                arr = np.pad(arr, (pad_len, 0), mode="edge")
            return arr

        w15 = pad(w15, win15)
        w1d = pad(w1d, win1d)
        wkp = pad(wkp, win1d)

        return {
            "price_15m": w15.tolist(),
            "price_1d": w1d.tolist(),
            "kimchi_premium": wkp.tolist(),
            "krw_balance": self.krw,
            "usdt_balance": self.usdt,
            "current_price": self._price(idx),
        }

    # ===== step =====
    def step(self, action):
        """
        Executes one environment step.
        - 모델은 현재 스텝(t)의 close 기준으로 action 결정
        - order_price는 같은 스텝(t)의 close 기준 ±α (모델이 낸 값)
        - reward는 다음 스텝(t+1) 가치 변화 기준
        """
        # --- 액션 파싱 ---
        if isinstance(action, dict):
            action_str = action.get("action")
            order_price = action.get("order_price")
        elif isinstance(action, (tuple, list)):
            action_str = action[0]
            order_price = action[1] if len(action) > 1 else None
        else:
            action_str = str(action)
            order_price = None

        # --- 현재 스텝(t)의 시세 ---
        ts_now = self.df_15m.index[self.step_idx]
        close_now = float(self._price(self.step_idx))
        prev_value = self._portfolio_value()

        # --- 모델이 낸 오더 프라이스: 같은 스텝 close 기준 ±δ ---
        # (없으면 현재 close로 대체)
        if order_price is None or not isinstance(order_price, (int, float)) or np.isnan(order_price):
            trade_price = close_now
        else:
            trade_price = float(order_price)

        # --- 거래 수행 (현재 스텝에서 체결됨) ---
        if action_str == "BUY" and self.krw > 0:
            qty = self.krw / trade_price
            self.usdt += qty * (1 - self.fee)
            self.krw = 0.0
        elif action_str == "SELL" and self.usdt > 0:
            krw_gain = self.usdt * trade_price * (1 - self.fee)
            self.krw += krw_gain
            self.usdt = 0.0
        # HOLD는 패스

        # --- 다음 스텝으로 진행 ---
        self.step_idx += 1
        done = self.step_idx >= self.max_steps - 1

        # --- reward는 다음봉 기준으로 계산 ---
        if done:
            reward = 0.0
            post_value = self._portfolio_value()
            delta_value = 0.0
        else:
            next_price = float(self._price(self.step_idx))
            post_value = self.krw + self.usdt * next_price
            delta_value = (post_value - prev_value) / max(prev_value, 1e-8)
            reward = np.tanh(delta_value * 5.0)

        # --- 상태 업데이트 ---
        next_state = self._state()

        # --- reward 계산 후 metrics 구성 ---
        next_price = float(self._price(self.step_idx))  # 다음 스텝 close

        metrics = {
            "ts_now": ts_now,
            "close_now": close_now,  # 현재 스텝의 close
            "price": next_price,  # ✅ 다음 스텝 close (reward용)
            "action": action_str,
            "order_price": trade_price,
            "value_prev": prev_value,
            "value_now": post_value,
            "delta_value": delta_value,
            "reward": reward,
            "krw_balance": self.krw,
            "usdt_balance": self.usdt,
        }
        return next_state, metrics, done

    def _portfolio_value(self):
        p = self._price(self.step_idx)
        return self.krw + self.usdt * p