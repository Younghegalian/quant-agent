import yaml
from core.agent_core import RLAgent
from live.live_loop import run_live
import random
from core.utils import load_config

# config.yaml 읽어서 window 크기 불러오기
cfg = load_config()
win15 = cfg["data"]["window_15m"]
win1d = cfg["data"]["window_1d"]

def dummy_get_live_state():
    """
    실전 입력 포맷과 동일하게 시계열 길이를 자동 조정한 더미 데이터 생성기.
    config.yaml의 data.window_15m / window_1d 값을 그대로 반영.
    """
    base_price = 1370.0

    # 최근 15분봉 close (길이: win15)
    prices_15m = [base_price + random.uniform(-1.5, 1.5) for _ in range(win15)]

    # 최근 일봉 close (길이: win1d)
    prices_1d = [base_price + random.uniform(-3.0, 3.0) for _ in range(win1d)]

    # 김치프리미엄 (길이: win1d)
    kimchi_premium = [random.uniform(1.5, 2.5) for _ in range(win1d)]

    return {
        "price_15m": prices_15m,
        "price_1d": prices_1d,
        "kimchi_premium": kimchi_premium,
        "krw_balance": random.uniform(900_000, 1_000_000),
        "usdt_balance": random.uniform(0, 10),
        "current_price": prices_15m[-1],
    }

def dummy_execute_action(action, state):
    # 예시 메트릭 (실전에서는 거래소 결과로 채워짐)
    price = state["current_price"]
    value_prev = state["krw_balance"] + state["usdt_balance"] * price
    # ... 실제 체결 결과 반영 ...
    value_now = value_prev  # 더미
    return {"value_prev": value_prev, "value_now": value_now}

if __name__ == "__main__":
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    agent = RLAgent(cfg)
    # agent.load("...pt")  # 원하면 불러오기
    run_live(agent, dummy_get_live_state, dummy_execute_action, cfg)