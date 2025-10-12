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

        # --- 윈도우 길이 ---
        win15 = cfg["data"]["window_15m"]
        win1d = cfg["data"]["window_1d"]

        # --- 공통 구간으로 자르기 (양쪽 tz 동일/정렬 가정) ---
        start = max(price_15m.index[0], price_1d.index[0])
        end = min(price_15m.index[-1], price_1d.index[-1])
        price_15m = price_15m.loc[start:end].copy()
        price_1d = price_1d.loc[start:end].copy()

        # --- 최소 길이 체크 ---
        if len(price_1d.index) < win1d:
            raise ValueError(f"일봉 길이({len(price_1d)}) < win1d({win1d})")
        if len(price_15m.index) < win15:
            raise ValueError(f"15분봉 길이({len(price_15m)}) < win15({win15})")

        # --- 시작 시점 계산 (두 윈도우를 모두 만족) ---
        # 1) 일봉 윈도우가 '완성된' 마지막 인덱스의 시각 (off-by-one 방지: win1d-1)
        first_usable_1d_ts = price_1d.index[win1d - 1]

        # 2) 15m에서 해당 시각 이상이면서, 15m 윈도우까지 확보 가능한 첫 시점
        #    (즉, i >= win15-1 이고 15m_ts[i] >= first_usable_1d_ts)
        i = int(price_15m.index.searchsorted(first_usable_1d_ts, side="left"))
        i = max(i, win15 - 1)  # 15m 윈도우 확보

        if i >= len(price_15m.index):
            raise ValueError("시작 오프셋이 15분봉 범위를 초과합니다. 데이터 구간/윈도우를 확인하세요.")

        self.start_offset = i

        # --- 데이터 배열화 ---
        self.df_15m = price_15m
        self.df_1d = price_1d
        self.data_15m = price_15m["close"].to_numpy(dtype=np.float32)
        self.data_1d = price_1d["close"].to_numpy(dtype=np.float32)
        self.kp_data = price_1d["kp"].to_numpy(dtype=np.float32) if "kp" in price_1d.columns else None

        # --- 남은 스텝 수 (start_offset 이후만 학습/시뮬레이션 가능) ---
        self.max_steps = len(self.data_15m) - self.start_offset
        if self.max_steps <= 0:
            raise ValueError("start_offset 이후 진행 가능한 15m 스텝이 없습니다.")

        print(f"[Simulator] len(15m)={len(self.data_15m)}, len(1d)={len(self.data_1d)}, "
              f"start_offset={self.start_offset}, max_steps={self.max_steps}")

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

        ts = self.df_15m.index[idx]
        i1d = max(0, self.df_1d.index.searchsorted(ts, side="right") - 1)

        w15 = self.data_15m[max(0, idx - win15):idx]
        w1d = self.data_1d[max(0, i1d - win1d):i1d]
        wkp = self.kp_data[max(0, i1d - win1d):i1d]

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

        # --- 모델이 낸 오더 프라이스: 같은 스텝 close 기준 ±δ ---
        if action_str != "HOLD":
            trade_price = float(order_price)

        # --- 거래 수행 (현재 스텝에서 체결됨) ---
        if action_str == "BUY" and self.krw > 0:
            qty = self.krw / trade_price
            self.usdt += qty * (1 - self.fee)
            self.krw = self.krw - self.usdt * trade_price
        elif action_str == "SELL" and self.usdt > 0:
            krw_gain = self.usdt * trade_price * (1 - self.fee)
            self.krw += krw_gain
            self.usdt = 0.0
        # HOLD는 패스

        metrics = {
            "krw_balance": self.krw,
            "usdt_balance": self.usdt,
        }

        # --- 다음 스텝으로 진행 ---
        self.step_idx += 1
        done = self.step_idx >= self.max_steps - 1

        # --- 상태 업데이트 ---
        next_state = self._state()

        return next_state, metrics, done

    def _portfolio_value(self):
        p = self._price(self.step_idx)
        return self.krw + self.usdt * p