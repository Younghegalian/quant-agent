import yaml
from core.agent_core import RLAgent
from sim.trainer import Trainer

if __name__ == "__main__":
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    agent = RLAgent(cfg)
    trainer = Trainer(agent, cfg)
    trainer.train(episodes=5)