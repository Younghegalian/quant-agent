from agent_core import RLAgent
import numpy as np

agent = RLAgent()

for step in range(50):
    # 가짜 환경: 랜덤 state + 보상
    state = np.random.randn(10)
    action, logits, value = agent.act(state)
    reward = np.random.randn() * 0.1
    done = (step % 10 == 0)

    next_state = np.random.randn(10)
    agent.store_transition((state, action, reward, next_state, done, logits, value))

    if step % 10 == 0:
        agent.learn()