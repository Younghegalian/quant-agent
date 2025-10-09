import yaml
from core.agent_core import RLAgent
from sim.trainer import Trainer

if __name__ == "__main__":
    # 설정 로드
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ✅ 시뮬 모드 에이전트 생성
    agent = RLAgent(cfg, mode="sim")

    # 트레이너 생성 및 학습 시작
    trainer = Trainer(agent, cfg)
    trainer.train(episodes=5)