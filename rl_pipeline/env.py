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
You are an ML researcher modifying a GPT training script to minimize val_bpb \
(validation bits per byte). Lower is better. Training runs for a fixed 5-minute \
time budget on a single GPU.

Rules:
- Only modify train.py (not prepare.py).
- The code must run without crashing (common failures: OOM, shape mismatch, missing imports).
- Think carefully, then output the edit.
"""

EDIT_FORMAT = """\
Output exactly ONE edit in this format. Copy lines from the file EXACTLY as they \
appear, including indentation. Do NOT add or remove leading spaces.

<<<<<<< SEARCH
(exact lines copied from the current file)
=======
(replacement lines)
>>>>>>> REPLACE
"""


def build_prompt(
    repo_path: str,
    history: list[dict] | None = None,
    best_bpb: float | None = None,
) -> str:
    """Build the prompt from current train.py + experiment history."""
    train_py = Path(repo_path) / "train.py"
    code = train_py.read_text()

    parts = [
        SYSTEM_PROMPT,
        "## Current train.py\n```python\n" + code + "\n```\n",
    ]

    if best_bpb is not None:
        parts.append(f"Current best val_bpb: {best_bpb:.6f}. Beat this.\n")

    if history:
        parts.append("## Experiment history (recent first)")
        for exp in history[-10:]:  # last 10 experiments
            desc = exp.get("description", "unknown change")
            if exp.get("crashed"):
                error = exp.get("error", "")
                parts.append(f"- {desc}: CRASHED ({error})")
            else:
                parts.append(f"- {desc}: val_bpb={exp.get('val_bpb', '?')}")
        parts.append("")

    parts.append("## Your task")
    parts.append("Propose ONE change to improve val_bpb. " + EDIT_FORMAT)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Response parsing — extract SEARCH/REPLACE blocks and apply them
# ---------------------------------------------------------------------------

def strip_thinking(response: str) -> str:
    """Remove <think>...</think> blocks from the response."""
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()


def parse_edits(response: str) -> list[tuple[str, str]]:
    """Extract (search, replace) pairs from the model response, deduplicated."""
    response = strip_thinking(response)
    # Strip markdown code fences (```python, ```, etc.)
    response = re.sub(r"```\w*\r?\n?", "", response)
    # Normalize \r\n → \n
    response = response.replace("\r\n", "\n")
    # Flexible marker matching:
    #   - 3-7 angle brackets/equals signs
    #   - optional trailing whitespace on marker lines
    #   - replacement section can be empty (for deletions)
    pattern = (
        r"<{3,7}\s*SEARCH[ \t]*\n"
        r"(.*?)\n"
        r"={3,7}[ \t]*\n"
        r"(.*?)"
        r">{3,7}\s*REPLACE"
    )
    matches = re.findall(pattern, response, re.DOTALL)
    # Deduplicate and strip trailing newline from replacement
    seen = set()
    unique = []
    for search, replace in matches:
        replace = replace.rstrip("\n")
        key = (search, replace)
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


def _strip_common_leading_whitespace(text: str) -> str:
    """Remove common leading whitespace from all lines (like textwrap.dedent)."""
    lines = text.splitlines()
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return text
    min_indent = min(len(l) - len(l.lstrip()) for l in non_empty)
    if min_indent == 0:
        return text
    return "\n".join(l[min_indent:] if len(l) >= min_indent else l for l in lines)


def apply_edits(file_path: str, edits: list[tuple[str, str]]) -> bool:
    """Apply SEARCH/REPLACE edits to a file.

    Tries exact match first.  If that fails, strips common leading whitespace
    from the SEARCH/REPLACE block and retries (handles model adding extra
    indentation in its response).

    Returns True if all edits applied.
    """
    content = Path(file_path).read_text()
    for search, replace in edits:
        if search in content:
            content = content.replace(search, replace, 1)
        else:
            # Retry with leading whitespace stripped
            stripped_search = _strip_common_leading_whitespace(search)
            stripped_replace = _strip_common_leading_whitespace(replace)
            if stripped_search in content:
                content = content.replace(stripped_search, stripped_replace, 1)
            else:
                return False
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
    prompt = build_prompt(repo_path, history, best_bpb)

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
