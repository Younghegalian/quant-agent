import yaml
import pandas as pd
from core.agent_core import RLAgent
from sim.trainer import Trainer

if __name__ == "__main__":
    # ===== ① 설정 로드 =====
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ===== ② 실데이터 로드 =====
    # 15분봉 & 일봉 CSV는 사전에 전처리되어 있다고 가정
    df_15m = pd.read_csv(cfg["data"]["path_15m"], parse_dates=["timestamp"], index_col="timestamp")
    df_1d  = pd.read_csv(cfg["data"]["path_1d"],  parse_dates=["timestamp"], index_col="timestamp")

    # ===== ③ Agent 생성 =====
    agent = RLAgent(cfg, mode="sim")

    # ===== ④ Trainer 생성 =====
    trainer = Trainer(agent, cfg, price_15m=df_15m, price_1d=df_1d)

    # ===== ⑤ 학습 or 평가 =====
    # eval_mode=False → 학습 + 백테스트
    # eval_mode=True  → 평가(학습 없음, 행동만)
    agent.eval_mode = False
    trainer.train(episodes=5, eval_mode=False)

    # 만약 학습 후 평가도 하고 싶다면:
    agent.eval_mode = True
    agent.load("your_model.pt")
    trainer.train(episodes=1, eval_mode=True)