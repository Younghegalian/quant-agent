import time
import random
from typing import Callable, Dict, Any
from core.utils import log, load_config

# ---- 전역 설정 (dummy 입력에 window 반영) ----
_cfg = load_config()
_win15 = _cfg["data"]["window_15m"]
_win1d = _cfg["data"]["window_1d"]

def dummy_get_live_state() -> Dict[str, Any]:
    """실전 입력 포맷과 동일한 더미 상태 (config의 window 길이 자동 반영)"""
    return {
        "price_15m": [random.uniform(1360, 1375) for _ in range(_win15)],
        "price_1d": [random.uniform(1360, 1375) for _ in range(_win1d)],
        "kimchi_premium": [random.uniform(0.01, 0.02) for _ in range(_win1d)],
        "krw_balance": random.uniform(900_000, 1_000_000),
        "usdt_balance": random.uniform(0, 10),
        "current_price": random.uniform(1360, 1375),
    }

def dummy_execute_action(action_str: str, state: Dict[str, Any]) -> Dict[str, float]:
    """
    실전에서는 거래소 API 체결 결과를 사용.
    여기서는 평가액이 약간 흔들린다고 가정한 더미 metrics만 반환.
    """
    price = float(state.get("current_price", 0.0))
    v_prev = float(state.get("krw_balance", 0.0) + state.get("usdt_balance", 0.0) * price)
    drift = random.uniform(-300, 300)
    v_now = v_prev + drift
    return {"value_prev": v_prev, "value_now": v_now, "price": price}

def run_live(
    agent,
    get_live_state: Callable[[], Dict[str, Any]],
    execute_action: Callable[[str, Dict[str, Any]], Dict[str, float]],
    cfg: Dict[str, Any],
):
    """
    실전(온라인) 루프. 리워드는 반드시 agent 내부에서 계산한다.
    """
    step = 0
    update_interval = cfg["training"].get("update_interval", 64)
    sleep_sec = cfg.get("live", {}).get("refresh_interval", 5)

    log("[LIVE] Online RL Loop start")
    while True:
        try:
            # 1) 상태 수집 → 2) 행동 결정
            state = get_live_state()
            action_str = agent.act(state)  # "HOLD" / "SIGNAL" 또는 최종 "BUY"/"SELL"/"HOLD"

            # 3) 행동 집행(체결) → metrics 획득
            metrics = execute_action(action_str, state)

            # 4) 보상 계산(항상 Agent 내부)
            reward = agent.compute_reward(metrics)

            # 5) 버퍼 저장 (실전에서는 next_state=None로 둬도 무방)
            agent.store_transition(state, action_str, reward, None, False)

            # 6) 행동/상태 로그
            krw = state.get("krw_balance", 0.0)
            usdt = state.get("usdt_balance", 0.0)
            price = state.get("current_price", 0.0)
            val_now = metrics.get("value_now", 0.0)

            log(
                f"[LIVE] step={step:05d} | "
                f"action={action_str} | price={price:.2f} | "
                f"KRW={krw:.0f} | USDT={usdt:.4f} | "
                f"value={val_now:.2f} | reward={reward:.6f}"
            )

            # 7) 일정 step마다 학습
            if len(getattr(agent, "buffer", [])) >= update_interval:
                loss = agent.learn()
                if loss is not None:
                    log(f"[LIVE] step={step:05d} | PPO update | loss={loss:.6f}")

            step += 1
            time.sleep(sleep_sec)

        except Exception as e:
            log(f"[LIVE] step={step:05d} | ERROR: {e}")
            time.sleep(2)
            continue