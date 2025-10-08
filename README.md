# **USDT-RL: Online Reinforcement Learning Engine for K-Premia**

> 해당 프로젝트는 **Python 3.10+** 기반의 **Online Reinforcement Learning (RL)** 에이전트 엔진으로,  
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
 ├── __init__.py                         
 ├── agent_core.py                       # RLAgent 정의 (act(), learn())
 ├── model.py                            # GRU + Attention + Dual Head 구조
 ├── ppo_core.py                         # PPO 학습 루프 (선택사항)
 ├── state_schema.py                     # 입력 전처리, normalization, feature shaping
 ├── utils.py                            # 로그, 스케줄러, 메트릭 계산 등
 └── config.yaml                         # 하이퍼파라미터 설정 파일
```

> 💡 *시뮬레이터(`env/`)는 별도로 구성 예정. RL 브레인은 환경과 독립적으로 작동*

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
──────────────────────────────────────────────────────────────────────────────────────
   15m / 1d / 김프 시계열 입력
         │
         ▼
  GRU Encoders ×3
         │
         ▼
  Attention Pooling
         │
         ▼
  Dense Fusion  ←  계좌 상태 + 현재가
         │
         ├─▶ Policy Head : Softmax(2)   # Action logits
         └─▶ Value Head  : Linear(1)    # State value (V)
───────────────────────────────────────────────────────────────────────────────────────
```

Light temporal encoding × contextual fusion → dual-head decision network.


---

---

## 🧩 학습 설정

| 파라미터 | 기본값 | 설명 |
|-----------|---------|------|
| γ (discount) | 0.995 | 장기보상 가중치 |
| λ (GAE) | 0.95 | advantage decay |
| clip | 0.2 | PPO clipping ratio |
| lr | 3e-4 | 학습률 |
| entropy_coef | 0.01 | 탐험도 보상 |
| value_coef | 0.5 | value loss 가중치 |
| temperature | 0.1 | 탐험률 조절용 softmax 온도 |

---

## 🚀 실행 방법

1. **시뮬레이터와 함께 사용**
   ```python
    from agent.agent_core import RLAgent
    agent = RLAgent()
    state = env.get_state()      # 시뮬레이터가 만든 상태
    action = agent.act(state)    # RL 정책으로 행동 결정
    env.step(action)             # 거래 수행
   ```

---

## 🧑‍💻 Author

- **이승엽** — RL 브레인 설계 및 연구

---

> _“A demon of capitalism, swimming the net and devouring volatility.”_
