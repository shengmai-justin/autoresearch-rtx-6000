"""
GRPO (Group Relative Policy Optimization).

Given a batch of trajectories grouped by prompt:
  1. Compute advantages within each group (reward - group_mean) / group_std
  2. Compute log-probabilities of each response under current policy
  3. Compute ratio = exp(log_prob_new - log_prob_old)
  4. Clipped policy gradient loss
  5. Update model weights

Reference: Shao et al., 2024 — DeepSeekMath
"""

import torch
import torch.nn.functional as F

import config


class GRPO:
    def __init__(self, model, tokenizer, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=0.01,
        )
        self.clip_low = config.GRPO_CLIP_LOW
        self.clip_high = config.GRPO_CLIP_HIGH
        self.kl_coeff = config.GRPO_KL_COEFF
        self.step_count = 0

    def _compute_log_probs(self, prompt, response, require_grad=False):
        """Compute mean log probability of response tokens given prompt.

        Applies the same chat template used during generation so token
        sequences match.  Uses joint tokenization to avoid boundary merge
        issues: tokenize prompt alone to get prompt_len, then tokenize the
        full string (prompt+response) so merge artifacts only affect the
        boundary and prompt_len is derived from the same tokenizer call.
        """
        # Apply the same chat template used during generation
        messages = [{"role": "user", "content": prompt}]
        templated_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )

        # Tokenize templated prompt separately to find where response starts
        prompt_enc = self.tokenizer(
            templated_prompt, return_tensors="pt", truncation=True,
            max_length=config.MAX_CONTEXT,
            add_special_tokens=True,
        )
        prompt_len = prompt_enc["input_ids"].shape[1]

        # Tokenize full text (templated prompt + response) jointly
        full_enc = self.tokenizer(
            templated_prompt + response, return_tensors="pt", truncation=True,
            max_length=config.MAX_CONTEXT,
            add_special_tokens=True,
        )
        input_ids = full_enc["input_ids"].to(self.model.device)
        total_len = input_ids.shape[1]
        response_len = total_len - prompt_len

        if response_len <= 0:
            print(f"  WARNING: response has 0 tokens after tokenization "
                  f"(prompt_len={prompt_len}, total_len={total_len}). Skipping.")
            return None

        ctx = torch.enable_grad() if require_grad else torch.no_grad()
        with ctx:
            outputs = self.model(input_ids)
            logits = outputs.logits

        # Only score the response tokens
        shift_logits = logits[:, prompt_len - 1:-1, :]
        shift_labels = input_ids[:, prompt_len:]
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

        # Use mean instead of sum to normalize across different response lengths
        return token_log_probs.mean()

    def update(self, trajectories):
        """
        GRPO update on a batch of trajectories.

        Trajectories should be collected with the SAME model checkpoint
        (before this update call). We compute advantages within the group
        and do a single gradient step.
        """
        if len(trajectories) < 2:
            return {"loss": 0.0, "note": "need >= 2 trajectories for GRPO"}

        self.model.train()

        # Bug fix #2: use correction=0 (biased std) to avoid NaN with small batches
        rewards = torch.tensor([t.reward for t in trajectories], dtype=torch.float32)
        mean_r = rewards.mean()
        std_r = rewards.std(correction=0).clamp(min=1e-8)
        advantages = (rewards - mean_r) / std_r

        # Compute old log probs (detached — these are from the collection policy)
        old_log_probs = []
        valid_indices = []
        for i, traj in enumerate(trajectories):
            lp = self._compute_log_probs(traj.prompt, traj.response, require_grad=False)
            if lp is not None:
                old_log_probs.append(lp.detach())  # Bug fix #1: explicit detach
                valid_indices.append(i)

        if not valid_indices:
            return {"loss": 0.0, "note": "no valid trajectories for log prob computation"}

        # Policy gradient step
        total_loss = torch.tensor(0.0, device=self.model.device, requires_grad=True)
        for idx, i in enumerate(valid_indices):
            traj = trajectories[i]
            new_lp = self._compute_log_probs(traj.prompt, traj.response, require_grad=True)
            if new_lp is None:
                continue

            ratio = torch.exp(new_lp - old_log_probs[idx])

            # Asymmetric clipping
            adv = advantages[i].to(self.model.device)
            if adv >= 0:
                clipped_ratio = ratio.clamp(max=1 + self.clip_high)
            else:
                clipped_ratio = ratio.clamp(min=1 - self.clip_low)

            loss = -torch.min(ratio * adv, clipped_ratio * adv)
            total_loss = total_loss + loss

        total_loss = total_loss / max(len(valid_indices), 1)
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.step_count += 1

        return {
            "loss": total_loss.item(),
            "mean_reward": mean_r.item(),
            "std_reward": std_r.item(),
            "step": self.step_count,
            "valid_trajectories": len(valid_indices),
        }

    def save_checkpoint(self, path: str):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step_count": self.step_count,
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.model.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.step_count = ckpt["step_count"]
