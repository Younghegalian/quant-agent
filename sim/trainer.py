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
        self.history = []  # <== 모든 스텝 누적 기록용

        # 로그 디렉토리 준비
        os.makedirs(cfg["training"]["save_dir"], exist_ok=True)
        self.episode_log_path = save_path(cfg["training"]["save_dir"],f"train_summary_{ts()}.csv")
        self.step_log_dir = cfg["training"]["save_dir"]

    def train(self, episodes, eval_mode=False):

        for ep in range(episodes):
            self.step_records = []
            start = time.time()
            state = self.env.reset()
            done = False
            ep_reward = 0.0
            ep_step_counter = 0
            init_value = self.env._portfolio_value()
            prev_value = self.env._portfolio_value()

            while not done:
                # === 1) 행동 결정 ===
                action = self.agent.act(state)

                # === 3) 환경 스텝 ===
                next_state, metrics, done = self.env.step(action)
                curr_value = self.env._portfolio_value()
                delta_value = curr_value - prev_value
                prev_value = curr_value
                ep_step_counter += 1

                # === 4) 리워드 ===
                reward = self.agent.compute_reward(state, action)
                ep_reward += reward

                # === 5) 버퍼 저장 ===
                self.agent.store_transition(state, action, reward, done, False)
                state = next_state

                # === 6) 주기 학습 ===
                loss = None
                if not eval_mode and self.agent.ready_to_learn():
                    loss = self.agent.learn()

                # === 7) 상세 로그 ===
                action_str = str(action.get("action"))
                order_price = action.get("order_price")

                log_msg = (
                    f"[{self.env.df_15m.index[self.env.step_idx-1]:%Y-%m-%d %H:%M}] step={ep_step_counter:05d} | "
                    f"ep={ep+1:<2d} | "
                    f"close={self.env._price(self.env.step_idx-1):>10.2f} | "
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

                # === 8) step 로그 저장 ===
                self.step_records.append({
                    "episode": ep + 1,
                    "step": ep_step_counter,
                    "timestamp": self.env.df_15m.index[self.env.step_idx-1],
                    "close": self.env._price(self.env.step_idx-1),
                    "order_price": order_price,
                    "action": action_str,
                    "reward": reward,
                    "delta_value": delta_value,
                    "krw": metrics["krw_balance"],
                    "usdt": metrics["usdt_balance"],
                    "loss": loss,
                    "total_asset": curr_value,
                    "profit": curr_value-init_value,
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
                f"model_ep{ep+1}_{ts()}.pt"
            )

            log_path = save_path(
                self.cfg["training"]["save_dir"],
                f"log_ep{ep+1}_{ts()}.csv"
            )

            self.agent.save(ckpt_path)
            pd.DataFrame(self.step_records).to_csv(log_path, index=False)

            end = time.time()
            print(f"time per episode: {end - start:.4f}s")

        # === 최종 로그 저장 ===
        pd.DataFrame(self.history).to_csv(self.episode_log_path, index=False)
        log(f"[SIM] Episode summary saved to {self.episode_log_path}")