"""
RL algorithms. Each module exposes:
  - class with update(model, tokenizer, trajectories) -> metrics dict
  - save_checkpoint(path) and load_checkpoint(path)
"""

from algos.grpo import GRPO
from algos.none import NoRL
from algos.ppo import PPO

ALGO_REGISTRY = {
    "grpo": GRPO,
    "ppo": PPO,
    "none": NoRL,
}


def build_algo(name: str, model, tokenizer, **kwargs):
    cls = ALGO_REGISTRY[name]
    return cls(model, tokenizer, **kwargs)
