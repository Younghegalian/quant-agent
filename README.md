# **USDT-RL: Online Reinforcement Learning Engine for K-Premia**

> 해당 프로젝트는 **Python 3.11** 기반의 **Online Reinforcement Learning (RL)** 에이전트 엔진으로,  
> 김치프리미엄(K-Premia) 및 환율 변동을 이용한 **USDT–KRW 단타 전략**을  
> 실시간으로 학습하고 적응하도록 설계되었습니다.  
>  
> 거래 환경(Environment, K-Premia 모듈)으로부터 상태(state)를 주기적으로 입력받아,  
> GRU + Attention 기반의 정책 신경망을 통해 행동(action)을 산출하고,  
> 그 결과를 실시간 보상(reward)으로 반영하여 지속적으로 업데이트합니다.  
>  
> 정책은 고정된 규칙 기반이 아니라,  
> **시장 변동성과 리스크 요인에 따라 실시간으로 재학습(adaptive fine-tuning)** 되는 구조입니다.

---

### ✳️ 주요 특징
- **Online RL Architecture:** 시뮬레이션과 실거래 중 모두 학습 가능  
- **Dual-Timescale Input:** 15분봉 / 일봉 / 김프 시계열 병렬 인코딩  
- **Attention-Based Fusion:** 시간적 중요도에 따른 동적 가중  
- **Policy / Value Head 분리:** Softmax(2) + Value(1) 구조  
- **Lightweight Integration:** 서버 없이 로컬 환경(env)과 직접 연동  

---

### 🧭 프로젝트 목적
- 김치프리미엄 및 환율의 동적 변화를 실시간으로 반영해  
  **단기 시장 왜곡(temporal arbitrage)** 을 포착하고 반응하는 강화학습 기반 의사결정 엔진 개발  
- 단순한 예측형 모델이 아니라,  
  **행동 기반 피드백 루프(action–reward–update)** 를 통해  
  **스스로 적응하는 거래 정책(self-adaptive trading policy)** 구현

---

## 📂 프로젝트 구조

```text
agent/
├── config.yaml                     # 공통 설정 (model/ppo/training/policy/data)
├── run_sim.py                      # 시뮬 학습 엔트리포인트
├── run_live.py                     # 실전(온라인 RL) 엔트리포인트
│
├── core/                           # 💠 에이전트 코어 (시뮬/실전 공통)
│   ├── agent_core.py               # RLAgent: act()/compute_reward()/learn() + 버퍼 관리
│   ├── model.py                    # GRU + Attention + Policy(2) / Value(1)
│   ├── ppo_core.py                 # PPO 클립 손실/엔트로피/값함수 + optimizer
│   ├── memory.py                   # 간단한 ReplayBuffer(rolling)
│   └── utils.py                    # log(), load_config(), save_path() 등 유틸
│
├── sim/                            # 🧠 시뮬레이션(훈련) 전용
│   ├── simulator.py                # 과거데이터 기반 가상체결 + 상태/metrics 반환
│   └── trainer.py                  # 시뮬 학습 루프(Agent↔Simulator 연결)
│
└── live/                           # ⚡ 실전 루프(피드/실행은 외부 모듈 주입)
    └── live_loop.py                # get_live_state/execute_action 주입형 온라인 RL 루프
```


---

## ⚙️ 주요 기능

### `/act`
- 입력: 최근 시세, 계좌 상태, 김프 등  
- 출력: `action (0=Hold, 1=Action)`  
- 사용처: 실전/시뮬 모두 동일

### `/learn`
- 입력: transition mini-batch  
- 처리: PPO 업데이트  
- 출력: 업데이트 로그 (loss, reward 평균 등)

### `/save`, `/load`
- 정책 버전 관리 (`/checkpoints/v1_0_0.pt` 등)

### `/set_hparams`
- 탐험률(temperature), lr, clip 등 런타임 조정

---

## 🧠 Model Architecture


```text
─────────────────────────────────────────────────────────
Input:
 ├─ 15m close series  → GRU + Attention Pooling
 ├─ 1d close series   → GRU + Attention Pooling
 ├─ 1d Kimchi Premium    → GRU + Attention Pooling
 ├─ Account State     → Dense Encoding (KRW, USDT)
 └─ Current Price     → Scalar Input

Fusion:
 └─ Concat + Dense →
      ├─ Policy Head → Softmax(2)  (Hold / Trade)
      └─ Value Head  → Linear(1)
─────────────────────────────────────────────────────────
```


## Action semantics
```text
0: HOLD

1: SIGNAL (→ 계좌 비중에 따라 BUY / SELL로 자동 해석)
```



---

## 🧩 학습 설정

| **Category**                 | **Key**                         | **Default / Example** | **Description**                |
| ---------------------------- | ------------------------------- | --------------------- | ------------------------------ |
| **Model**                    | `model.hidden_dim`              | `64`                  | GRU hidden state 크기            |
|                              | `model.gru_layers`              | `1`                   | GRU layer 수                    |
|                              | `model.dropout`                 | `0.1`                 | Dropout 비율                     |
|                              | `model.attention`               | `true`                | Attention pooling 사용 여부        |
|                              | `model.action_dim`              | `2`                   | 행동 차원 (`BUY`, `SELL`)          |
| **PPO (공통)**                 | `ppo.gamma`                     | `0.99`                | Discount factor (보상 감쇠율)       |
|                              | `ppo.batch_size`                | `32`                  | 미니배치 크기                        |
| **Training (공통)**            | `training.device`               | `"cuda"`              | 연산 디바이스 (`cpu`/`cuda`)         |
|                              | `training.buffer_maxlen`        | `10000`               | 경험 버퍼 최대 크기                    |
|                              | `training.save_dir`             | `"checkpoints/"`      | 모델 저장 경로                       |
|                              | `training.save_interval_steps`  | `500`                 | 체크포인트 저장 주기(step 단위)           |
| **Training – Simulation**    | `training.sim.lr`               | `3e-4`                | 시뮬레이션 학습률                      |
|                              | `training.sim.clip_epsilon`     | `0.2`                 | PPO 클리핑 한계                     |
|                              | `training.sim.entropy_coef`     | `0.01`                | 엔트로피(탐험) 가중치                   |
|                              | `training.sim.value_coef`       | `0.5`                 | Value 손실 가중치                   |
|                              | `training.sim.update_epochs`    | `3`                   | 학습 반복(epoch) 수                 |
|                              | `training.sim.update_interval`  | `64`                  | 업데이트 주기 (버퍼 길이 기준)             |
|                              | `training.sim.update_freq`      | `5`                   | 학습 호출 간격 (step 기준)             |
| **Training – Live (Online)** | `training.live.lr`              | `1e-4`                | 실전용 학습률                        |
|                              | `training.live.clip_epsilon`    | `0.1`                 | PPO 클리핑 한계                     |
|                              | `training.live.entropy_coef`    | `0.02`                | 엔트로피(탐험) 가중치                   |
|                              | `training.live.value_coef`      | `0.4`                 | Value 손실 가중치                   |
|                              | `training.live.update_epochs`   | `1`                   | 온라인 업데이트 반복 수                  |
|                              | `training.live.update_interval` | `16`                  | 업데이트 주기 (버퍼 길이 기준)             |
|                              | `training.live.update_freq`     | `3`                   | 학습 호출 간격 (step 기준)             |
| **Live Execution**           | `live.refresh_interval`         | `10`                  | 실시간 데이터 업데이트 간격(초)             |
| **Policy Logic**             | `policy.signal_sell_threshold`  | `0.10`                | USDT 잔고 비중이 10% 이상이면 `SELL` 신호 |
| **Data Window**              | `data.window_15m`               | `15`                  | 15분봉 입력 시퀀스 길이                 |
|                              | `data.window_1d`                | `10`                  | 일봉 입력 시퀀스 길이                   |
| **Simulation Env**           | `sim.max_steps`                 | `1000`                | 시뮬 최대 step 수                   |
|                              | `sim.fee`                       | `0.001`               | 거래 수수료 비율                      |
|                              | `sim.init_krw`                  | `1000000`             | 초기 자본 (KRW)                    |
| **Versioning**               | `versioning.save_dir`           | `"checkpoints/"`      | 체크포인트 저장 경로                    |
|                              | `versioning.auto_timestamp`     | `true`                | 타임스탬프별 폴더 자동 생성 여부             |



---

## 🚀 실행 방법

1. **Simulated Training**
   ```python
   python run_sim.py
   ```
2. **Live (Online RL)**
   ```python
   python run_live.py
   ```

---

## 🧑‍💻 Author

- **이승엽** — RL 브레인 설계 및 연구

---

> _“A demon of capitalism, swimming the net and devouring volatility.”_
