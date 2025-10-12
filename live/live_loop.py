from datetime import datetime
import time
import random
import os
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

def dummy_execute_action(action_str: str, state: Dict[str, Any], order_price: float = None) -> Dict[str, float]:
    """
    실전에서는 거래소 API 체결 결과를 반환.
    order_price: agent.act()에서 지정한 주문가격 (+1/-1 보정 반영)
    """
    if order_price is None:
        order_price = state.get("current_price", 0.0)

    # 가상 체결 로직


    return {"krw_balance": random.uniform(900_000, 1_000_000),
            "usdt_balance": random.uniform(0, 10),
            "current_price": random.uniform(1360, 1375),}


def run_live(
    agent,
    get_live_state: Callable[[], Dict[str, Any]],
    execute_action: Callable[[str, Dict[str, Any], float], Dict[str, float]],
    cfg: Dict[str, Any],
):
    """
    실전(온라인) 루프. 리워드는 반드시 agent 내부에서 계산한다.
    """
    step = 0
    sleep_sec = cfg.get("live", {}).get("refresh_interval", 5)
    save_every = cfg.get("live", {}).get("save_interval_updates", 0)
    save_prefix = cfg.get("live", {}).get("save_prefix", "live_model")
    update_count = 0

    log("[LIVE] Online RL Loop start")

    while True:
        try:
            # 1) 상태 수집
            state = get_live_state()

            # 2) 행동 결정 (가격 포함)
            act_out = agent.act(state)  # {"action": "BUY"/"SELL"/"HOLD", "order_price": float}
            action_str = act_out["action"]
            order_price = act_out["order_price"]

            # 3) 행동 집행(체결) → metrics 획득
            metrics = execute_action(action_str, state, order_price)

            # 4) 자산 상태 반영
            price = float(state.get("current_price", 0.0))
            krw = float(state.get("krw_balance", 0.0))
            usdt = float(state.get("usdt_balance", 0.0))
            metrics["krw_balance"] = krw
            metrics["usdt_balance"] = usdt
            metrics["price"] = price

            # 5) 보상 계산
            reward = agent.compute_reward(metrics, action_str)

            # 6) 딜레이
            time.sleep(sleep_sec)

            # 7) 버퍼 저장
            agent.store_transition(state, action_str, reward, None, False)

            # 8) 로그 출력
            val_now = krw + usdt * price
            order_str = f"{order_price:.2f}" if order_price is not None else "None"
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            log(
                f"[{now_str}] [LIVE] step={step:05d} | "
                f"action={action_str} | order_price={order_str} | "
                f"market_price={price:.2f} | "
                f"KRW={krw:.0f} | USDT={usdt:.4f} | "
                f"value={val_now:.2f} | reward={reward:.6f}"
            )

            # 9) 일정 step마다 PPO 학습
            if agent.ready_to_learn():
                loss = agent.learn()
                if loss is not None:
                    update_count += 1
                    log(f"[{now_str}] [LIVE] step={step:05d} | PPO update #{update_count} | loss={loss:.6f}")

                    # ✅ 주기적 모델 저장
                    if save_every > 0 and update_count % save_every == 0:
                        save_path = os.path.join(
                            agent.save_dir, f"{save_prefix}_upd{update_count:04d}.pth"
                        )
                        agent.save(save_path)
                        log(f"[{now_str}] [LIVE] model checkpoint saved → {save_path}")

            step += 1

        except Exception as e:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log(f"[{now_str}] [LIVE] step={step:05d} | ERROR: {e}")
            time.sleep(2)
            continue