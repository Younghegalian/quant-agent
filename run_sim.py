import pandas as pd
from core.utils import load_config
from core.agent_core import RLAgent
from sim.trainer import Trainer

if __name__ == "__main__":
    # ===== ① 설정 로드 =====
    cfg = load_config()
    win15 = cfg["data"]["window_15m"]
    win1d = cfg["data"]["window_1d"]

    # ===== ② 실데이터 로드 =====
    # 15분봉
    df_15m = pd.read_csv(
        "io/USDT_15m_20251010.csv",
        usecols=["timestamp_kst", "close"],
        parse_dates=["timestamp_kst"]
    )
    df_15m.rename(columns={"timestamp_kst": "timestamp"}, inplace=True)
    df_15m.set_index("timestamp", inplace=True)
    df_15m = df_15m[["close"]]
    df_15m.index = df_15m.index.tz_localize(None)

    # 일봉
    df_1d = pd.read_csv(
        "io/USDT_kimchi_days_20251010.csv",
        usecols=["timestamp_kst", "upbit_usdt_krw", "kimchi_premium(%)"],
        parse_dates=["timestamp_kst"]
    )
    df_1d.rename(
        columns={
            "timestamp_kst": "timestamp",
            "upbit_usdt_krw": "close",
            "kimchi_premium(%)": "kp"
        },
        inplace=True
    )
    df_1d.set_index("timestamp", inplace=True)
    df_1d = df_1d[["close", "kp"]]
    df_1d.index = df_1d.index.tz_localize(None)

    # ===== ③ Agent 생성 =====
    agent = RLAgent(cfg, mode="sim")

    # ===== ④ Trainer 생성 =====
    trainer = Trainer(agent, cfg, price_15m=df_15m, price_1d=df_1d)

    # ===== ⑤ 학습 or 평가 =====
    agent.eval_mode = False
    trainer.train(episodes=20, eval_mode=False)

    # ===== ⑥ 학습 후 평가 =====
    #agent.eval_mode = True
    #agent.load("your_model.pt")
    #trainer.train(episodes=1, eval_mode=True)