"""
Environment: run one episode and return a trajectory with reward.

An episode:
  1. Build prompt (current train.py + history + instruction)
  2. Model generates response (reasoning + code edit)
  3. Apply edit to train.py
  4. Run train.py on the eval GPU
  5. Parse val_bpb from output
  6. Compute reward
  7. Reset repo state
"""

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import config


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    prompt: str = ""
    response: str = ""
    reward: float = 0.0
    val_bpb: float = 0.0
    peak_vram_mb: float = 0.0
    crashed: bool = False
    edit_applied: bool = False
    episode_time: float = 0.0
    metadata: dict = field(default_factory=dict)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            return cls(**json.load(f))


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an ML researcher modifying a GPT training script to achieve the lowest \
possible val_bpb (validation bits per byte). Lower is better.

You can change anything in train.py: model architecture, optimizer, hyperparameters, \
batch size, model size, etc. The training runs for a fixed 5-minute time budget.

Rules:
- Only modify train.py. Do not touch prepare.py.
- The code must run without crashing.
- Simpler changes that improve val_bpb are preferred over complex ones.
"""

EDIT_FORMAT = """\
Output your reasoning first, then your edit in this exact format:

<<<<<<< SEARCH
(exact lines from the current file to find)
=======
(replacement lines)
>>>>>>> REPLACE

You may include multiple SEARCH/REPLACE blocks if needed.
"""


def build_prompt(repo_path: str, history: list[dict] | None = None) -> str:
    """Build the prompt from current train.py + experiment history."""
    train_py = Path(repo_path) / "train.py"
    code = train_py.read_text()

    parts = [
        SYSTEM_PROMPT,
        "## Current train.py\n```python\n" + code + "\n```\n",
    ]

    if history:
        parts.append("## Experiment history (recent first)")
        for exp in history[-10:]:  # last 10 experiments
            status = "CRASH" if exp.get("crashed") else f"val_bpb={exp.get('val_bpb', '?')}"
            parts.append(f"- {exp.get('description', 'unknown')}: {status}")
        parts.append("")

    parts.append("## Your task")
    parts.append("Propose ONE change to improve val_bpb. " + EDIT_FORMAT)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Response parsing — extract SEARCH/REPLACE blocks and apply them
# ---------------------------------------------------------------------------

def parse_edits(response: str) -> list[tuple[str, str]]:
    """Extract (search, replace) pairs from the model response."""
    pattern = r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE"
    matches = re.findall(pattern, response, re.DOTALL)
    return matches


def apply_edits(file_path: str, edits: list[tuple[str, str]]) -> bool:
    """Apply SEARCH/REPLACE edits to a file. Returns True if all edits applied."""
    content = Path(file_path).read_text()
    for search, replace in edits:
        if search not in content:
            return False
        content = content.replace(search, replace, 1)
    Path(file_path).write_text(content)
    return True


# ---------------------------------------------------------------------------
# Run train.py and parse results
# ---------------------------------------------------------------------------

def run_training(repo_path: str) -> dict:
    """Run train.py and parse the output. Returns dict with val_bpb, peak_vram_mb, etc."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(config.TRAIN_GPU)

    result = subprocess.run(
        ["uv", "run", "train.py"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=config.EVAL_TIME_BUDGET + 120,  # generous timeout
        env=env,
    )

    output = result.stdout + result.stderr

    if result.returncode != 0:
        return {"crashed": True, "error": output[-2000:]}

    # Parse val_bpb and peak_vram_mb from output
    parsed = {}
    for line in output.splitlines():
        if line.startswith("val_bpb:"):
            parsed["val_bpb"] = float(line.split(":")[1].strip())
        elif line.startswith("peak_vram_mb:"):
            parsed["peak_vram_mb"] = float(line.split(":")[1].strip())

    if "val_bpb" not in parsed:
        return {"crashed": True, "error": "val_bpb not found in output"}

    parsed["crashed"] = False
    return parsed


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

def compute_reward(val_bpb: float | None, crashed: bool) -> float:
    """Simple reward: improvement over baseline. Negative for crashes."""
    if crashed:
        return config.CRASH_PENALTY
    return config.BASELINE_BPB - val_bpb


# ---------------------------------------------------------------------------
# Git management
# ---------------------------------------------------------------------------

def git_save(repo_path: str, message: str):
    """Commit current state."""
    subprocess.run(["git", "add", "train.py"], cwd=repo_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", message], cwd=repo_path, capture_output=True)


def git_reset(repo_path: str):
    """Reset train.py to last commit. Raises on failure to avoid corrupted state."""
    result = subprocess.run(
        ["git", "checkout", "train.py"],
        cwd=repo_path, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git_reset failed: {result.stderr}")


def git_get_short_hash(repo_path: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repo_path, capture_output=True, text=True,
    )
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Run one episode
# ---------------------------------------------------------------------------

def run_episode(
    generate_fn,
    repo_path: str,
    history: list[dict] | None = None,
    best_bpb: float | None = None,
    keep_if_improved: bool = False,
) -> Trajectory:
    """
    Run one full episode.

    Args:
        generate_fn: callable(prompt) -> response string.
        repo_path: path to the autoresearch repo.
        history: list of past experiment dicts.
        best_bpb: current best val_bpb (for sequential mode).
        keep_if_improved: if True, keep changes that beat best_bpb
                          and commit them (autoresearch mode).
                          if False, always reset (RL training mode).

    Returns:
        Trajectory with prompt, response, reward, and metadata.
    """
    t0 = time.time()

    # 1. Build prompt
    prompt = build_prompt(repo_path, history)

    # 2. Generate response
    response = generate_fn(prompt)

    # 3. Parse and apply edits
    edits = parse_edits(response)
    if not edits:
        return Trajectory(
            prompt=prompt,
            response=response,
            reward=config.CRASH_PENALTY * 0.5,  # less harsh than crash
            crashed=True,
            edit_applied=False,
            episode_time=time.time() - t0,
            metadata={"error": "no edits found in response"},
        )

    train_py_path = os.path.join(repo_path, "train.py")
    applied = apply_edits(train_py_path, edits)
    if not applied:
        git_reset(repo_path)
        return Trajectory(
            prompt=prompt,
            response=response,
            reward=config.CRASH_PENALTY * 0.5,
            crashed=True,
            edit_applied=False,
            episode_time=time.time() - t0,
            metadata={"error": "SEARCH block not found in train.py"},
        )

    # 4. Run train.py (catch timeout so repo is always cleaned up)
    try:
        result = run_training(repo_path)
    except subprocess.TimeoutExpired:
        result = {"crashed": True, "error": "train.py timed out"}

    # 5. Compute reward
    crashed = result.get("crashed", False)
    val_bpb = result.get("val_bpb", 0.0)
    reward = compute_reward(val_bpb if not crashed else None, crashed)

    # 6. Keep or reset
    improved = (not crashed) and (best_bpb is None or val_bpb < best_bpb)
    kept = False

    if keep_if_improved and improved:
        # Sequential mode: keep the improvement, commit it
        description = response[:100].replace("\n", " ")
        git_save(repo_path, f"experiment: val_bpb={val_bpb:.6f} | {description}")
        kept = True
    else:
        # RL mode or no improvement: always reset
        git_reset(repo_path)

    return Trajectory(
        prompt=prompt,
        response=response,
        reward=reward,
        val_bpb=val_bpb,
        peak_vram_mb=result.get("peak_vram_mb", 0.0),
        crashed=crashed,
        edit_applied=True,
        episode_time=time.time() - t0,
        metadata={**result, "kept": kept, "improved": improved},
    )
