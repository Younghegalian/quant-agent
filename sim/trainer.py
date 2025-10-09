from .simulator import Simulator
from core.utils import log, save_path, ts

class Trainer:
    def __init__(self, agent, cfg, price_15m=None):
        self.agent = agent
        self.env = Simulator(cfg, price_15m=price_15m)
        self.cfg = cfg

    def train(self, episodes=10):
        step_counter = 0
        for ep in range(episodes):
            state = self.env.reset()
            done = False
            ep_reward = 0.0
            ep_step_counter = 0  # 에피소드별 스텝 카운터 추가

            while not done:
                action = self.agent.act(state)  # "BUY"/"SELL"/"HOLD"
                next_state, metrics, done = self.env.step(action)
                reward = self.agent.compute_reward(metrics)

                self.agent.store_transition(state, action, reward, next_state, done, aux=metrics)
                state = next_state
                ep_reward += reward
                step_counter += 1
                ep_step_counter += 1

                if self.agent.ready_to_learn():
                    loss = self.agent.learn()
                    # loss가 None이 아닐 때만 로그 출력 (불필요한 로그 방지)
                    if loss is not None and step_counter % 50 == 0:
                        log(f"[SIM] step={step_counter} loss={loss:.6f} ep_reward={ep_reward:.4f}")

                # 주기적 저장: 파일명에 ep, step 추가하여 중복 방지 및 추적 용이
                if step_counter % self.cfg["training"]["save_interval_steps"] == 0:
                    path = save_path(
                        self.cfg["training"]["save_dir"],
                        f"sim_ep{ep+1}_step{step_counter}_{ts()}.pt"
                    )
                    self.agent.save(path)

            # 에피소드별 로그 출력
            log(f"[SIM] Episode {ep+1}/{episodes} total_reward={ep_reward:.4f} steps={ep_step_counter}")
