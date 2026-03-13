"""
Main RL training loop.

Usage (on cloud server):
    python rl_train.py
    python rl_train.py --algo none                    # ablation: no RL
    python rl_train.py --algo none --keep-if-improved  # ablation: real autoresearch loop
    python rl_train.py --resume checkpoints/step_50.pt
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from env import run_episode, run_training, git_reset, Trajectory
from algos import build_algo


def setup_model():
    """Load model and tokenizer from local directory."""
    print(f"Loading model from {config.MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_DIR,
        torch_dtype=torch.bfloat16,
        device_map=f"cuda:{config.VLLM_GPU}",
    )
    return model, tokenizer


def make_generate_fn(model, tokenizer):
    """Create the generate function used by env.run_episode."""
    def generate(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                temperature=config.TEMPERATURE,
                do_sample=True,
                top_p=0.95,
            )
        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)
    return generate


def run_baseline(repo_path: str) -> float:
    """Run unmodified train.py once to establish baseline val_bpb."""
    print("Running baseline train.py to establish initial val_bpb...")
    git_reset(repo_path)  # ensure clean state
    result = run_training(repo_path)
    if result.get("crashed"):
        raise RuntimeError(f"Baseline run crashed: {result.get('error', 'unknown')}")
    baseline = result["val_bpb"]
    print(f"Baseline val_bpb: {baseline:.6f}")
    return baseline


def log_step(step, metrics, trajectories, log_file):
    """Append step results to log file."""
    entry = {
        "step": step,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        **metrics,
        "episodes": [
            {
                "val_bpb": t.val_bpb,
                "reward": t.reward,
                "crashed": t.crashed,
                "edit_applied": t.edit_applied,
                "episode_time": round(t.episode_time, 1),
            }
            for t in trajectories
        ],
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"Step {step}: loss={metrics.get('loss', 0):.4f} "
          f"mean_reward={metrics.get('mean_reward', 0):.4f} "
          f"val_bpbs={[round(t.val_bpb, 4) for t in trajectories if not t.crashed]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default=config.ALGO, choices=["grpo", "ppo", "none"])
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    parser.add_argument("--steps", type=int, default=config.NUM_STEPS)
    parser.add_argument("--episodes", type=int, default=config.EPISODES_PER_STEP)
    parser.add_argument("--keep-if-improved", action="store_true", default=config.KEEP_IF_IMPROVED,
                        help="Keep changes that improve val_bpb (sequential autoresearch mode)")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline run (use BASELINE_BPB from config)")
    args = parser.parse_args()

    # Setup
    model, tokenizer = setup_model()
    algo = build_algo(args.algo, model, tokenizer)
    generate_fn = make_generate_fn(model, tokenizer)

    # Resume from checkpoint — restore step count for correct file naming
    start_step = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        algo.load_checkpoint(args.resume)
        start_step = algo.step_count
        print(f"Resuming from step {start_step}")

    # Establish baseline val_bpb for reward computation
    if not args.skip_baseline:
        config.BASELINE_BPB = run_baseline(config.REPO_PATH)
    print(f"Using BASELINE_BPB: {config.BASELINE_BPB:.6f}")

    # Dirs
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.TRAJECTORY_DIR, exist_ok=True)
    log_file = "training_log.jsonl"

    # History of experiments (shared across episodes)
    history = []
    best_bpb = config.BASELINE_BPB  # start from baseline

    if args.algo == "none":
        total_episodes = args.steps * args.episodes  # treat total as flat count
        print(f"Starting ablation (no RL): {total_episodes} episodes, "
              f"keep_if_improved={args.keep_if_improved}")
        print(f"LLM on GPU {config.VLLM_GPU}, train.py on GPU {config.TRAIN_GPU}")

        all_trajectories = []
        for ep in range(total_episodes):
            print(f"  Episode {ep + 1}/{total_episodes}...", flush=True)
            t0 = time.time()
            traj = run_episode(
                generate_fn, config.REPO_PATH, history,
                best_bpb=best_bpb,
                keep_if_improved=args.keep_if_improved,
            )
            all_trajectories.append(traj)

            if not traj.crashed and traj.val_bpb > 0:
                if traj.val_bpb < best_bpb:
                    best_bpb = traj.val_bpb
                    print(f"    New best val_bpb: {best_bpb:.6f}")

            history.append({
                "val_bpb": traj.val_bpb,
                "crashed": traj.crashed,
                "kept": traj.metadata.get("kept", False),
                "description": traj.response[:200] if traj.response else "",
            })

            traj.save(os.path.join(
                config.TRAJECTORY_DIR,
                f"ep{ep:04d}.json",
            ))

            # Log each episode individually
            metrics = algo.update([traj])
            metrics["episode_time"] = time.time() - t0
            metrics["best_bpb"] = best_bpb
            log_step(ep, metrics, [traj], log_file)

    else:
        print(f"Starting RL training: algo={args.algo}, steps={args.steps}, "
              f"episodes/step={args.episodes}, keep_if_improved={args.keep_if_improved}")
        print(f"LLM on GPU {config.VLLM_GPU}, train.py on GPU {config.TRAIN_GPU}")

        for step in range(start_step, start_step + args.steps):
            t0 = time.time()

            # Collect episodes
            trajectories = []
            for ep in range(args.episodes):
                print(f"  Step {step}, episode {ep}/{args.episodes}...", flush=True)
                traj = run_episode(
                    generate_fn, config.REPO_PATH, history,
                    best_bpb=best_bpb,
                    keep_if_improved=args.keep_if_improved,
                )
                trajectories.append(traj)

                if not traj.crashed and traj.val_bpb > 0:
                    if traj.val_bpb < best_bpb:
                        best_bpb = traj.val_bpb
                        print(f"    New best val_bpb: {best_bpb:.6f}")

                history.append({
                    "val_bpb": traj.val_bpb,
                    "crashed": traj.crashed,
                    "kept": traj.metadata.get("kept", False),
                    "description": traj.response[:200] if traj.response else "",
                })

                traj.save(os.path.join(
                    config.TRAJECTORY_DIR,
                    f"step{step:04d}_ep{ep:02d}.json",
                ))

            # RL update
            metrics = algo.update(trajectories)
            metrics["step_time"] = time.time() - t0
            metrics["best_bpb"] = best_bpb

            # Regenerate generate_fn with updated model weights
            generate_fn = make_generate_fn(model, tokenizer)

            # Log
            log_step(step, metrics, trajectories, log_file)

            # Checkpoint
            if (step + 1) % config.CHECKPOINT_INTERVAL == 0:
                ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"step_{step + 1}.pt")
                algo.save_checkpoint(ckpt_path)
                print(f"  Saved checkpoint: {ckpt_path}")

    print(f"Training complete. Best val_bpb: {best_bpb:.6f}")


if __name__ == "__main__":
    main()
