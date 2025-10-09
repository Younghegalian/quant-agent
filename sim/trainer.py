from .simulator import Simulator
from core.utils import log, save_path, ts
import pandas as pd
import os

class Trainer:
    """
    실데이터 기반 백테스트용 Trainer
    - 기존 구조 유지 (Simulator, Agent, Trainer 그대로)
    - Agent가 보상(reward) 계산 담당
    - 각 episode 단위로 결과 기록 및 모델 저장
    """

    def __init__(self, agent, cfg, price_15m=None, price_1d=None):
        self.agent = agent
        self.env = Simulator(cfg, price_15m=price_15m, price_1d=price_1d)
        self.cfg = cfg
        self.history = []  # episode 기록 저장용

        # 저장 디렉토리 준비
        os.makedirs(cfg["training"]["save_dir"], exist_ok=True)
        self.log_path = os.path.join(cfg["training"]["save_dir"], "train_log.csv")

    def train(self, episodes=1, eval_mode=False):
        """
        백테스트 학습 루프
        - episodes: 백테스트 반복 횟수
        - eval_mode: True면 학습(update) 없이 행동만 수행
        """
        total_steps = 0

        for ep in range(episodes):
            state = self.env.reset()
            done = False
            ep_reward = 0.0
            ep_step_counter = 0
            init_value = self.env._portfolio_value()

            while not done:
                # ---- 행동 수행 ----
                action = self.agent.act(state)
                next_state, metrics, done = self.env.step(action)
                reward = self.agent.compute_reward(metrics)

                # transition 저장
                self.agent.store_transition(
                    state, action, reward, next_state, done, aux=metrics
                )

                # 상태 업데이트
                state = next_state
                ep_reward += reward
                total_steps += 1
                ep_step_counter += 1

                # ---- 학습 (평가모드면 skip) ----
                if not eval_mode and self.agent.ready_to_learn():
                    loss = self.agent.learn()
                    if loss is not None and total_steps % 50 == 0:
                        log(f"[SIM] step={total_steps} loss={loss:.6f} ep_reward={ep_reward:.4f}")

                # ---- 주기적 저장 ----
                if total_steps % self.cfg["training"]["save_interval_steps"] == 0:
                    path = save_path(
                        self.cfg["training"]["save_dir"],
                        f"ep{ep+1}_step{total_steps}_{ts()}.pt"
                    )
                    self.agent.save(path)

            # ---- 에피소드 종료 후 기록 ----
            final_value = self.env._portfolio_value()
            delta_value = final_value - init_value
            self.history.append({
                "episode": ep + 1,
                "steps": ep_step_counter,
                "total_reward": ep_reward,
                "init_value": init_value,
                "final_value": final_value,
                "profit": delta_value,
            })

            log(f"[SIM] Episode {ep+1}/{episodes} "
                f"reward={ep_reward:.4f} Δvalue={delta_value:.2f} steps={ep_step_counter}")

            # ---- 체크포인트 저장 ----
            ckpt_path = save_path(
                self.cfg["training"]["save_dir"],
                f"final_ep{ep+1}_{ts()}.pt"
            )
            self.agent.save(ckpt_path)

        # ---- 모든 episode 결과 CSV로 저장 ----
        pd.DataFrame(self.history).to_csv(self.log_path, index=False)
        log(f"[SIM] Training finished. Log saved at {self.log_path}")