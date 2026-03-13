"""
Evaluate a model checkpoint by running N episodes and reporting stats.

Usage:
    python rl_evaluate.py                          # evaluate base model
    python rl_evaluate.py --checkpoint checkpoints/step_50.pt
    python rl_evaluate.py --episodes 20            # more episodes for better stats
"""

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from env import run_episode
from rl_train import make_generate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {config.MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_DIR,
        torch_dtype=torch.bfloat16,
        device_map=f"cuda:{config.VLLM_GPU}",
    )

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=model.device)
        model.load_state_dict(ckpt["model_state_dict"])

    generate_fn = make_generate_fn(model, tokenizer)

    # Run episodes
    results = []
    history = []
    for i in range(args.episodes):
        print(f"Episode {i + 1}/{args.episodes}...", flush=True)
        traj = run_episode(generate_fn, config.REPO_PATH, history)
        results.append(traj)
        history.append({
            "val_bpb": traj.val_bpb,
            "crashed": traj.crashed,
        })

    # Report
    valid = [r for r in results if not r.crashed]
    crashed = [r for r in results if r.crashed]

    print("\n=== Evaluation Results ===")
    print(f"Total episodes: {len(results)}")
    print(f"Crashed: {len(crashed)}")
    print(f"Valid: {len(valid)}")
    if valid:
        bpbs = [r.val_bpb for r in valid]
        rewards = [r.reward for r in valid]
        print(f"val_bpb  — mean: {sum(bpbs)/len(bpbs):.6f}, "
              f"best: {min(bpbs):.6f}, worst: {max(bpbs):.6f}")
        print(f"reward   — mean: {sum(rewards)/len(rewards):.4f}")

    # Save
    tag = "base" if not args.checkpoint else os.path.basename(args.checkpoint).replace(".pt", "")
    out_path = f"eval_{tag}.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps({
                "val_bpb": r.val_bpb,
                "reward": r.reward,
                "crashed": r.crashed,
                "episode_time": r.episode_time,
            }) + "\n")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
