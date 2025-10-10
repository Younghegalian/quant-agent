from .simulator import Simulator
from core.utils import log, save_path, ts
import pandas as pd
import os
import time
import numpy as np

class Trainer:
    """
    강화학습 백테스트용 Trainer (전체 스텝 로그 + Δvalue 세밀 기록)
    - action/order_price는 t 시점
    - reward는 t+1 결과 기준
    """

    def __init__(self, agent, cfg, price_15m=None, price_1d=None):
        self.agent = agent
        self.env = Simulator(cfg, price_15m=price_15m, price_1d=price_1d)
        self.cfg = cfg
        self.history = []
        self.step_records = []  # <== 모든 스텝 누적 기록용

        # 로그 디렉토리 준비
        os.makedirs(cfg["training"]["save_dir"], exist_ok=True)
        self.episode_log_path = os.path.join(cfg["training"]["save_dir"], "train_log.csv")
        self.step_log_path = os.path.join(cfg["training"]["save_dir"], "train_log_steps.csv")

    def train(self, episodes=1, eval_mode=False):
        total_steps = 0

        for ep in range(episodes):
            start = time.time()
            state = self.env.reset()
            done = False
            ep_reward = 0.0
            ep_step_counter = 0
            init_value = self.env._portfolio_value()

            while not done:
                # === 1) 행동 결정 ===
                action = self.agent.act(state)

                # === 2) 현재 상태 ===
                ts_now = self.env.df_15m.index[self.env.step_idx]
                close_now = self.env._price(self.env.step_idx)
                prev_value = self.env._portfolio_value()

                # === 3) 환경 스텝 ===
                next_state, metrics, done = self.env.step(action)

                # === 4) 리워드 ===
                reward = self.agent.compute_reward(metrics, action)
                curr_value = metrics.get("value_now", self.env._portfolio_value())
                delta_value = curr_value - prev_value

                # === 5) 버퍼 저장 ===
                self.agent.store_transition(state, action, reward, next_state, done, aux=metrics)

                # === 6) 업데이트 ===
                state = next_state
                ep_reward += reward
                total_steps += 1
                ep_step_counter += 1

                # === 7) 학습 ===
                loss = None
                if not eval_mode and self.agent.ready_to_learn():
                    loss = self.agent.learn()

                # === 8) 액션 파싱 ===
                if isinstance(action, dict):
                    action_str = str(action.get("action"))
                    order_price = action.get("order_price")
                elif isinstance(action, (tuple, list)):
                    action_str = str(action[0])
                    order_price = action[1] if len(action) > 1 else None
                else:
                    action_str = str(action)
                    order_price = None

                # === 9) 상세 로그 ===
                log_msg = (
                    f"[{ts_now:%Y-%m-%d %H:%M}] step={total_steps:05d} | "
                    f"ep={ep+1:<2d} | "
                    f"close={close_now:>10.2f} | "
                    f"action={action_str:<5}"
                )
                if order_price is not None:
                    log_msg += f" | order_price={order_price:>10.2f}"
                else:
                    log_msg += f" | order_price={'None':>10}"

                log_msg += (
                    f" | reward={reward:>10.6f}"
                    f" | Δvalue={delta_value:>12.6f}"
                    f" | value_now={curr_value:>12.2f}"
                    f" | KRW={metrics['krw_balance']:>12.2f}"
                    f" | USDT={metrics['usdt_balance']:>10.6f}"
                )
                if loss is not None:
                    log_msg += f" | loss={loss:>12.6e}"

                # print(log_msg, flush=True)

                # === 10) step 로그 저장 ===
                self.step_records.append({
                    "episode": ep + 1,
                    "step": total_steps,
                    "timestamp": ts_now,
                    "close": close_now,
                    "action": action_str,
                    "order_price": order_price,
                    "reward": reward,
                    "delta_value": delta_value,
                    "value": curr_value,
                    "krw": metrics["krw_balance"],
                    "usdt": metrics["usdt_balance"],
                    "loss": loss,
                })

                # time.sleep(0.1)

            # === 에피소드 종료 ===
            final_value = self.env._portfolio_value()
            delta_value_ep = final_value - init_value
            self.history.append({
                "episode": ep + 1,
                "steps": ep_step_counter,
                "total_reward": ep_reward,
                "init_value": init_value,
                "final_value": final_value,
                "profit": delta_value_ep,
            })

            log(f"[SIM] Episode {ep+1}/{episodes} reward={ep_reward:.4f} Δvalue={delta_value_ep:.2f}")

            ckpt_path = save_path(
                self.cfg["training"]["save_dir"],
                f"final_ep{ep+1}_{ts()}.pt"
            )
            self.agent.save(ckpt_path)

            end = time.time()
            print(f"time per episode: {end - start:.4f}s")

        # === 전체 로그 저장 ===
        pd.DataFrame(self.history).to_csv(self.episode_log_path, index=False)
        pd.DataFrame(self.step_records).to_csv(self.step_log_path, index=False)
        log(f"[SIM] All step logs saved to {self.step_log_path}")
        log(f"[SIM] Episode summary saved to {self.episode_log_path}")