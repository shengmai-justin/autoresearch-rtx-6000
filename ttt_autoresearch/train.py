"""
TTT-Discover RL training loop for autoresearch.

Main loop: PUCT selects parent → generate GROUP_SIZE rollouts →
evaluate in parallel on multiple GPUs via Ray →
entropic adaptive beta advantages → importance sampling loss +
KL penalty → update LoRA weights.
"""
from __future__ import annotations

import argparse
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
# Config (defaults, overridable via CLI args)
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="TTT-Discover RL for autoresearch")
    p.add_argument("--model-dir", default="./models/Qwen3.5-27B")
    p.add_argument("--llm-gpu", type=int, default=6)
    p.add_argument("--eval-gpus", type=int, nargs="+", default=[7])
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=4e-5)
    p.add_argument("--kl-coef", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=32768)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--puct-c", type=float, default=1.0)
    p.add_argument("--puct-max-buffer", type=int, default=500)
    p.add_argument("--puct-topk-children", type=int, default=2)
    p.add_argument("--repo-path", default="..")
    p.add_argument("--log-dir", default="./ttt_log")
    p.add_argument("--no-overlap", action="store_true", help="Disable gen/eval overlap")
    return p.parse_args()


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
# Logging
# ---------------------------------------------------------------------------

def _append_jsonl(path: str, obj: dict):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def _log_eval_result(result: dict, eval_time: float):
    if result["success"]:
        print(f"    eval: val_bpb={result['val_bpb']:.6f} ({eval_time:.1f}s)")
    else:
        print(f"    eval: FAILED: {result['output'][:100]} ({eval_time:.1f}s)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    overlap = not args.no_overlap

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.join(args.log_dir, "rollouts"), exist_ok=True)
    code_dir = os.path.abspath(os.path.dirname(__file__))
    rollout_log = os.path.join(args.log_dir, "rollouts.jsonl")

    print("=" * 60)
    print("TTT-Discover Autoresearch")
    print(f"  eval_gpus={args.eval_gpus}  overlap={overlap}")
    print(f"  steps={args.num_steps}  group_size={args.group_size}  batch_size={args.batch_size}")
    print("=" * 60)

    # -- Init Ray + eval workers ---------------------------------------------
    ray.init(ignore_reinit_error=True)
    workers = [
        EvalWorker.remote(gpu, args.repo_path, i, code_dir)
        for i, gpu in enumerate(args.eval_gpus)
    ]
    print(f"Created {len(workers)} eval workers")

    # -- Load model + LoRA ---------------------------------------------------
    print(f"Loading model from {args.model_dir} on GPU {args.llm_gpu}...")
    model, tokenizer = load_model(
        args.model_dir, gpu_id=args.llm_gpu,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
    )

    # -- Baseline run --------------------------------------------------------
    print("Running baseline train.py...")
    from env import run_training
    baseline_result = run_training(args.repo_path, gpu_id=args.eval_gpus[0])
    if baseline_result.get("crashed"):
        print(f"ERROR: Baseline train.py crashed: {baseline_result.get('error', '')}")
        sys.exit(1)
    baseline_bpb = baseline_result["val_bpb"]
    print(f"Baseline val_bpb: {baseline_bpb:.6f}")

    # -- Read original train.py code -----------------------------------------
    original_code = Path(os.path.join(args.repo_path, "train.py")).read_text()

    # -- Initialize PUCT sampler ---------------------------------------------
    initial_state = State(
        timestep=0,
        code=original_code,
        value=-baseline_bpb,
        observation=baseline_result.get("output", ""),
    )
    sampler = PUCTSampler(
        initial_state=initial_state,
        log_dir=args.log_dir,
        puct_c=args.puct_c,
        max_buffer=args.puct_max_buffer,
        topk_children=args.puct_topk_children,
    )

    best_bpb = baseline_bpb
    step_log = []

    # -- Main loop -----------------------------------------------------------
    for step in range(args.num_steps):
        step_start = time.time()
        print(f"\n{'='*60}")
        print(f"Step {step}/{args.num_steps} | Best val_bpb: {best_bpb:.6f} | Buffer: {sampler.buffer_size()}")
        print(f"{'='*60}")

        parents = [sampler.sample_state() for _ in range(args.batch_size)]
        all_episodes: list[tuple[State, list[dict]]] = []
        step_gen_times = []
        step_eval_times = []

        for pi, parent in enumerate(parents):
            print(f"\n  Parent {pi}: val_bpb={-parent.value:.6f}" if parent.value is not None else f"\n  Parent {pi}: no value")
            prompt = build_prompt(parent)
            parent_dict = parent.to_dict()

            # Generate rollouts + submit evals to Ray workers
            rollouts = []
            for g in range(args.group_size):
                print(f"    Rollout {g+1}/{args.group_size}...", end=" ", flush=True)
                gen_start = time.time()

                text, full_ids, old_logprobs, prompt_len = generate_with_logprobs(
                    model, tokenizer, prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
                gen_time = time.time() - gen_start
                num_new_tokens = len(full_ids) - prompt_len
                step_gen_times.append(gen_time)
                print(f"generated ({num_new_tokens} tokens, {gen_time:.1f}s)", flush=True)

                worker = workers[g % len(workers)]
                ref = worker.evaluate.remote(parent_dict, text, step)
                rollouts.append({
                    "full_ids": full_ids,
                    "old_logprobs": old_logprobs,
                    "prompt_len": prompt_len,
                    "response_text": text,
                    "num_new_tokens": num_new_tokens,
                    "gen_time": gen_time,
                    "eval_ref": ref,
                    "eval_start": time.time(),
                })
                if not overlap:
                    ray.get(ref)

            # Collect all eval results
            episodes = []
            for g, r in enumerate(rollouts):
                result = ray.get(r["eval_ref"])
                eval_time = time.time() - r["eval_start"]
                step_eval_times.append(eval_time)
                if result.get("child_state") is not None:
                    result["child_state"] = State.from_dict(result["child_state"])
                _log_eval_result(result, eval_time)
                if result["success"] and result["val_bpb"] < best_bpb:
                    best_bpb = result["val_bpb"]
                    print(f"    *** NEW BEST: {best_bpb:.6f} ***")
                    child = result["child_state"]
                    if child is not None:
                        Path(os.path.join(args.log_dir, "best_train.py")).write_text(child.code)

                episodes.append({
                    "full_ids": r["full_ids"],
                    "old_logprobs": r["old_logprobs"],
                    "prompt_len": r["prompt_len"],
                    "result": result,
                    "reward": result["reward"],
                })

                _append_jsonl(rollout_log, {
                    "step": step,
                    "parent_id": pi,
                    "rollout_id": g,
                    "success": result["success"],
                    "val_bpb": result.get("val_bpb"),
                    "reward": result["reward"],
                    "num_tokens": r["num_new_tokens"],
                    "gen_time_s": round(r["gen_time"], 1),
                    "eval_time_s": round(eval_time, 1),
                    "error": result["output"][:200] if not result["success"] else None,
                })

                rollout_path = os.path.join(
                    args.log_dir, "rollouts", f"step{step:04d}_p{pi}_r{g}.txt"
                )
                with open(rollout_path, "w") as f:
                    f.write(r["response_text"])

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
        all_ratios = []
        all_kls = []

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
                    temperature=args.temperature,
                )

                ratio = torch.exp(new_lp - old_lp)
                all_ratios.append(ratio.detach().cpu())
                loss = -(ratio * adv.to(model.device)).mean()

                if args.kl_coef > 0:
                    base_lp = compute_base_logprobs(
                        model, tokenizer, full_ids, plen,
                        temperature=args.temperature,
                    )
                    kl = (new_lp - base_lp).mean()
                    all_kls.append(kl.item())
                    loss = loss + args.kl_coef * kl

                loss.backward()
                total_loss += loss.item() * len(new_lp)
                num_tokens += len(new_lp)

        if num_tokens > 0:
            clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.max_grad_norm,
            )
            optimizer.step()
            avg_loss = total_loss / num_tokens
            print(f"\n  RL update: avg_loss={avg_loss:.4f}, tokens={num_tokens}")
        else:
            print("\n  RL update: skipped (no valid episodes)")

        # -- Save checkpoint + metrics ---------------------------------------
        sampler.save(step)
        step_time = time.time() - step_start

        all_rewards = [ep["reward"] for _, eps in all_episodes for ep in eps]
        all_bpbs = [ep["result"]["val_bpb"] for _, eps in all_episodes for ep in eps if ep["result"]["success"]]
        n_success = sum(1 for _, eps in all_episodes for ep in eps if ep["result"]["success"])
        n_total = sum(len(eps) for _, eps in all_episodes)

        ratio_stats = {}
        if all_ratios:
            cat_ratios = torch.cat(all_ratios)
            ratio_stats = {
                "ratio/mean": round(cat_ratios.mean().item(), 4),
                "ratio/min": round(cat_ratios.min().item(), 4),
                "ratio/max": round(cat_ratios.max().item(), 4),
            }

        step_info = {
            "step": step,
            "best_bpb": best_bpb,
            "buffer_size": sampler.buffer_size(),
            "avg_loss": round(total_loss / max(num_tokens, 1), 6),
            "num_tokens": num_tokens,
            "step_time_s": round(step_time, 1),
            "success_rate": f"{n_success}/{n_total}",
            "rewards": [round(r, 4) for r in all_rewards],
            "val_bpbs": [round(b, 6) for b in all_bpbs],
            "avg_gen_time_s": round(sum(step_gen_times) / max(len(step_gen_times), 1), 1),
            "avg_eval_time_s": round(sum(step_eval_times) / max(len(step_eval_times), 1), 1),
            "kl_mean": round(sum(all_kls) / max(len(all_kls), 1), 6) if all_kls else None,
            **ratio_stats,
        }
        step_log.append(step_info)
        print(f"  Step time: {step_time/60:.1f} min | Success: {n_success}/{n_total}")

        with open(os.path.join(args.log_dir, "step_log.json"), "w") as f:
            json.dump(step_log, f, indent=2)

    # -- Done ----------------------------------------------------------------
    ray.shutdown()
    print(f"\n{'='*60}")
    print(f"Training complete. Best val_bpb: {best_bpb:.6f}")
    best = sampler.best_state()
    if best:
        Path(os.path.join(args.log_dir, "best_train.py")).write_text(best.code)
        print(f"Best code saved to {args.log_dir}/best_train.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
