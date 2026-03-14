"""
TTT-Discover RL training loop for autoresearch.

Main loop: PUCT selects parent → generate GROUP_SIZE rollouts →
evaluate in parallel on multiple GPUs via Ray →
entropic adaptive beta advantages → importance sampling loss +
KL penalty → update LoRA weights.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import ray
import torch
from torch.nn.utils import clip_grad_norm_
from pathlib import Path

# Local imports (same directory)
from puct import State, PUCTSampler
from env import build_prompt, create_worker_repo
from model import (
    load_model,
    generate_with_logprobs,
    compute_response_logprobs,
    compute_base_logprobs,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_DIR = "./models/Qwen3.5-27B"
LLM_GPU = 6
EVAL_GPUS = [7]  # GPUs for train.py evaluation (add more to parallelize)
LORA_RANK = 32
LORA_ALPHA = 64

NUM_STEPS = 50
GROUP_SIZE = 4          # rollouts per parent
BATCH_SIZE = 1          # parents per step
LEARNING_RATE = 4e-5
KL_PENALTY_COEF = 0.1
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 32768
MAX_GRAD_NORM = 1.0

PUCT_C = 1.0
PUCT_MAX_BUFFER = 500
PUCT_TOPK_CHILDREN = 2

REPO_PATH = ".."        # autoresearch repo (parent dir)
LOG_DIR = "./ttt_log"

# When True, start generating next rollout while current eval runs on another GPU.
# When False, wait for eval to finish before generating next rollout.
OVERLAP_GEN_EVAL = True


# ---------------------------------------------------------------------------
# Ray eval worker
# ---------------------------------------------------------------------------

@ray.remote
class EvalWorker:
    """Each worker owns an isolated repo copy and a GPU."""

    def __init__(self, gpu_id: int, base_repo: str, worker_id: int, code_dir: str):
        import sys as _sys
        _sys.path.insert(0, code_dir)
        self.gpu_id = gpu_id
        from env import create_worker_repo
        self.repo_path = create_worker_repo(base_repo, worker_id)

    def evaluate(self, parent_dict: dict, response_text: str, step: int) -> dict:
        from puct import State
        from env import evaluate_episode
        parent = State.from_dict(parent_dict)
        result = evaluate_episode(
            self.repo_path, parent, response_text,
            gpu_id=self.gpu_id, step=step,
        )
        # Serialize State for Ray transport
        if result.get("child_state") is not None:
            result["child_state"] = result["child_state"].to_dict()
        return result


# ---------------------------------------------------------------------------
# Entropic adaptive beta advantages (from discover/ttt_discover/rl/train.py:103-155)
# ---------------------------------------------------------------------------

def compute_entropic_advantages(rewards: list[float]) -> torch.Tensor:
    """
    Compute LOO entropic advantages with adaptive beta.
    Binary search for beta s.t. KL(q_beta || uniform) ≈ log(2).
    """
    r = torch.tensor(rewards, dtype=torch.float32)
    k = r.shape[0]

    if k < 2:
        return torch.zeros_like(r)

    delta = math.log(2)
    beta_max = 1e6
    iters = 60
    eps = 1e-12
    logK = math.log(k)

    def kl_hat(beta_scalar: float) -> float:
        b = torch.tensor(beta_scalar, dtype=r.dtype)
        logits = b * (r - r.max())
        logq = logits - torch.logsumexp(logits, dim=0)
        q = torch.exp(logq)
        kl = (q * (logq + logK)).sum()
        return float(kl.item())

    lo, hi = 0.0, 1.0
    if kl_hat(hi) < delta:
        while hi < beta_max and kl_hat(hi) < delta:
            hi *= 2.0
        if kl_hat(hi) < delta:
            beta = hi
        else:
            beta = None
    else:
        beta = None

    if beta is None:
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if kl_hat(mid) < delta:
                lo = mid
            else:
                hi = mid
        beta = hi

    e = torch.exp(beta * (r - r.max()))
    if k == 1:
        Z = e
    else:
        Z = (e.sum() - e) / (k - 1)
    w = e / (Z + eps)
    return w - 1.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    code_dir = os.path.abspath(os.path.dirname(__file__))

    print("=" * 60)
    print("TTT-Discover Autoresearch")
    print(f"  eval_gpus={EVAL_GPUS}  overlap={OVERLAP_GEN_EVAL}")
    print("=" * 60)

    # -- Init Ray + eval workers ---------------------------------------------
    ray.init(ignore_reinit_error=True)
    workers = [
        EvalWorker.remote(gpu, REPO_PATH, i, code_dir)
        for i, gpu in enumerate(EVAL_GPUS)
    ]
    print(f"Created {len(workers)} eval workers")

    # -- Load model + LoRA ---------------------------------------------------
    print(f"Loading model from {MODEL_DIR} on GPU {LLM_GPU}...")
    model, tokenizer = load_model(
        MODEL_DIR, gpu_id=LLM_GPU,
        lora_rank=LORA_RANK, lora_alpha=LORA_ALPHA,
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE, betas=(0.9, 0.95), eps=1e-8,
    )

    # -- Baseline run --------------------------------------------------------
    print("Running baseline train.py...")
    from env import run_training
    baseline_result = run_training(REPO_PATH, gpu_id=EVAL_GPUS[0])
    if baseline_result.get("crashed"):
        print(f"ERROR: Baseline train.py crashed: {baseline_result.get('error', '')}")
        sys.exit(1)
    baseline_bpb = baseline_result["val_bpb"]
    print(f"Baseline val_bpb: {baseline_bpb:.6f}")

    # -- Read original train.py code -----------------------------------------
    original_code = Path(os.path.join(REPO_PATH, "train.py")).read_text()

    # -- Initialize PUCT sampler ---------------------------------------------
    initial_state = State(
        timestep=0,
        code=original_code,
        value=-baseline_bpb,
        observation=baseline_result.get("output", ""),
    )
    sampler = PUCTSampler(
        initial_state=initial_state,
        log_dir=LOG_DIR,
        puct_c=PUCT_C,
        max_buffer=PUCT_MAX_BUFFER,
        topk_children=PUCT_TOPK_CHILDREN,
    )

    best_bpb = baseline_bpb
    step_log = []

    # -- Main loop -----------------------------------------------------------
    for step in range(NUM_STEPS):
        step_start = time.time()
        print(f"\n{'='*60}")
        print(f"Step {step}/{NUM_STEPS} | Best val_bpb: {best_bpb:.6f} | Buffer: {sampler.buffer_size()}")
        print(f"{'='*60}")

        parents = [sampler.sample_state() for _ in range(BATCH_SIZE)]
        all_episodes: list[tuple[State, list[dict]]] = []

        for pi, parent in enumerate(parents):
            print(f"\n  Parent {pi}: val_bpb={-parent.value:.6f}" if parent.value is not None else f"\n  Parent {pi}: no value")
            prompt = build_prompt(parent)
            parent_dict = parent.to_dict()

            # Generate rollouts + submit evals to Ray workers
            rollouts = []
            for g in range(GROUP_SIZE):
                print(f"    Rollout {g+1}/{GROUP_SIZE}...", end=" ", flush=True)
                gen_start = time.time()

                text, full_ids, old_logprobs, prompt_len = generate_with_logprobs(
                    model, tokenizer, prompt,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                )
                gen_time = time.time() - gen_start
                print(f"generated ({len(full_ids)-prompt_len} tokens, {gen_time:.1f}s)", flush=True)

                worker = workers[g % len(workers)]
                ref = worker.evaluate.remote(parent_dict, text, step)
                rollouts.append({
                    "full_ids": full_ids,
                    "old_logprobs": old_logprobs,
                    "prompt_len": prompt_len,
                    "eval_ref": ref,
                    "eval_start": time.time(),
                })
                # Sequential mode: block until this eval finishes
                if not OVERLAP_GEN_EVAL:
                    ray.get(ref)

            # Collect all eval results
            episodes = []
            for r in rollouts:
                result = ray.get(r["eval_ref"])
                eval_time = time.time() - r["eval_start"]
                # Deserialize child state
                if result.get("child_state") is not None:
                    result["child_state"] = State.from_dict(result["child_state"])
                _log_eval_result(result, eval_time)
                if result["success"] and result["val_bpb"] < best_bpb:
                    best_bpb = result["val_bpb"]
                    print(f"    *** NEW BEST: {best_bpb:.6f} ***")
                episodes.append({
                    "full_ids": r["full_ids"],
                    "old_logprobs": r["old_logprobs"],
                    "prompt_len": r["prompt_len"],
                    "result": result,
                    "reward": result["reward"],
                })

            all_episodes.append((parent, episodes))

        # -- Update PUCT tree ------------------------------------------------
        for parent, episodes in all_episodes:
            for ep in episodes:
                child = ep["result"].get("child_state")
                if child is not None:
                    sampler.update_state(child, parent)
                else:
                    sampler.record_failed_rollout(parent)

        # -- RL training step ------------------------------------------------
        optimizer.zero_grad()
        total_loss = 0.0
        num_tokens = 0

        for parent, episodes in all_episodes:
            rewards = [ep["reward"] for ep in episodes]
            advantages = compute_entropic_advantages(rewards)

            for ep, adv in zip(episodes, advantages):
                if abs(adv.item()) < 1e-8:
                    continue

                full_ids = ep["full_ids"]
                old_lp = ep["old_logprobs"].to(model.device)
                plen = ep["prompt_len"]

                new_lp = compute_response_logprobs(
                    model, tokenizer, full_ids, plen,
                    temperature=TEMPERATURE,
                )

                ratio = torch.exp(new_lp - old_lp)
                loss = -(ratio * adv.to(model.device)).mean()

                if KL_PENALTY_COEF > 0:
                    base_lp = compute_base_logprobs(
                        model, tokenizer, full_ids, plen,
                        temperature=TEMPERATURE,
                    )
                    kl = (new_lp - base_lp).mean()
                    loss = loss + KL_PENALTY_COEF * kl

                loss.backward()
                total_loss += loss.item() * len(new_lp)
                num_tokens += len(new_lp)

        if num_tokens > 0:
            clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                MAX_GRAD_NORM,
            )
            optimizer.step()
            avg_loss = total_loss / num_tokens
            print(f"\n  RL update: avg_loss={avg_loss:.4f}, tokens={num_tokens}")
        else:
            print("\n  RL update: skipped (no valid episodes)")

        # -- Save checkpoint -------------------------------------------------
        sampler.save(step)
        step_time = time.time() - step_start

        step_info = {
            "step": step,
            "best_bpb": best_bpb,
            "buffer_size": sampler.buffer_size(),
            "avg_loss": total_loss / max(num_tokens, 1),
            "num_tokens": num_tokens,
            "step_time_s": step_time,
        }
        step_log.append(step_info)
        print(f"  Step time: {step_time/60:.1f} min")

        with open(os.path.join(LOG_DIR, "step_log.json"), "w") as f:
            json.dump(step_log, f, indent=2)

    # -- Done ----------------------------------------------------------------
    ray.shutdown()
    print(f"\n{'='*60}")
    print(f"Training complete. Best val_bpb: {best_bpb:.6f}")
    best = sampler.best_state()
    if best:
        best_code_path = os.path.join(LOG_DIR, "best_train.py")
        Path(best_code_path).write_text(best.code)
        print(f"Best code saved to {best_code_path}")
    print(f"{'='*60}")


def _log_eval_result(result: dict, eval_time: float):
    if result["success"]:
        print(f"    eval: val_bpb={result['val_bpb']:.6f} ({eval_time:.1f}s)")
    else:
        print(f"    eval: FAILED: {result['output'][:100]} ({eval_time:.1f}s)")


if __name__ == "__main__":
    main()
