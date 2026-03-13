"""
PPO (Proximal Policy Optimization).

TODO: Implement when needed. Same interface as GRPO:
  - __init__(model, tokenizer, **kwargs)
  - update(trajectories) -> metrics dict
  - save_checkpoint(path)
  - load_checkpoint(path)

PPO differs from GRPO by using a learned value function (critic)
to estimate advantages, rather than group-relative normalization.
"""

import torch
import config


class PPO:
    def __init__(self, model, tokenizer, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.step_count = 0
        # TODO: initialize value head, optimizer, etc.
        raise NotImplementedError("PPO not yet implemented. Use 'grpo' or 'none'.")

    def update(self, trajectories):
        raise NotImplementedError

    def save_checkpoint(self, path: str):
        raise NotImplementedError

    def load_checkpoint(self, path: str):
        raise NotImplementedError
