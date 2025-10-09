from core.agent_core import RLAgent
from core.utils import load_config
from live.live_loop import run_live, dummy_get_live_state, dummy_execute_action

if __name__ == "__main__":
    cfg = load_config()
    agent = RLAgent(cfg)
    run_live(agent, dummy_get_live_state, dummy_execute_action, cfg)