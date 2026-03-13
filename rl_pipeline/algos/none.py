"""
No-RL ablation baseline.
Collects trajectories but never updates the model.
Used to measure how the base model performs without any RL training.
"""


class NoRL:
    def __init__(self, model, tokenizer, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.step_count = 0

    def update(self, trajectories):
        self.step_count += 1
        rewards = [t.reward for t in trajectories]
        return {
            "loss": 0.0,
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "note": "no-rl baseline, no weight update",
            "step": self.step_count,
        }

    def save_checkpoint(self, path: str):
        pass  # nothing to save

    def load_checkpoint(self, path: str):
        pass  # nothing to load
