import time
from typing import Callable, Dict, Any

from core.utils import log
import random
from core.utils import load_config

cfg = load_config()
win15 = cfg["data"]["window_15m"]
win1d = cfg["data"]["window_1d"]

def dummy_get_live_state():
    return {
        "price_15m": [random.uniform(1360, 1375) for _ in range(win15)],
        "price_1d": [random.uniform(1360, 1375) for _ in range(win1d)],
        "kimchi_premium": [random.uniform(0.01, 0.02) for _ in range(win1d)],
        "krw_balance": random.uniform(900_000, 1_000_000),
        "usdt_balance": random.uniform(0, 10),
        "current_price": random.uniform(1360, 1375),
    }

def run_live(
        agent,
        get_live_state: Callable[[], Dict[str, Any]],
        execute_action: Callable[[str, Dict[str, Any]], Dict[str, float]],
        cfg: Dict[str, Any]
):
    step = 0
    update_interval = cfg["training"]["update_interval"]

    while True:
        # ✅ 환경에서 상태 관측, 행동 선택, 행동 실행, 보상 계산, transition 저장
        try:
            state = get_live_state()
            action = agent.act(state)
            metrics = execute_action(action, state)
            reward = agent.compute_reward(metrics)
            agent.store_transition(state, action, reward, None, False)

        # TODO: 자세한 에러핸들링 필요
        except Exception as e:
            log(f"[LIVE] step={step:05d} | Error: {e}")
            continue

        # ✅ 행동 로그 출력
        krw = state.get("krw_balance", 0)
        usdt = state.get("usdt_balance", 0)
        price = state.get("current_price", 0)
        value_now = metrics.get("value_now", 0)

        log(
            f"[LIVE] step={step:05d} | "
            f"action={action} | price={price:.2f} | "
            f"KRW={krw:.0f} | USDT={usdt:.3f} | "
            f"value={value_now:.2f} | reward={reward:.6f}"
        )

        # ✅ 일정 step마다 학습
        if len(agent.buffer) >= update_interval:
            loss = agent.learn()
            log(f"[LIVE] step={step:05d} | PPO update | loss={loss:.6f}")

        step += 1
        time.sleep(cfg["training"].get("live_step_interval", 5))  # 예: 5초 간격