import yaml
from core.agent_core import RLAgent
from live.live_loop import run_live, dummy_get_live_state, dummy_execute_action

if __name__ == "__main__":
    # 설정 로드
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ✅ 실전 모드 에이전트 생성
    agent = RLAgent(cfg, mode="live")

    # ✅ 실전 루프 시작
    run_live(agent, dummy_get_live_state, dummy_execute_action, cfg)