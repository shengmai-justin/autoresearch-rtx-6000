"""
Environment functions: prompt building, edit parsing, train.py execution.

Adapted from rl_pipeline/env.py for State-based prompts.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from puct import State


# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an ML researcher modifying a GPT training script to minimize val_bpb \
(validation bits per byte). Lower is better. Training runs for a fixed 5-minute \
time budget on a single GPU.

Rules:
- Only modify train.py (not prepare.py).
- The code must run without crashing (common failures: OOM, shape mismatch, missing imports).
- Think carefully, then output the edit."""

EDIT_FORMAT = """\
Output exactly ONE edit in this format. Copy lines from the file EXACTLY as they \
appear, including indentation. Do NOT add or remove leading spaces.

<<<<<<< SEARCH
(exact lines copied from the current file)
=======
(replacement lines)
>>>>>>> REPLACE"""


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_prompt(state: State) -> str:
    """Build prompt from system prompt + state context."""
    parts = [
        SYSTEM_PROMPT,
        state.to_prompt(),
        "## Your task",
        "Propose ONE change to improve val_bpb. " + EDIT_FORMAT,
    ]
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Edit parsing (from rl_pipeline/env.py:121-149)
# ---------------------------------------------------------------------------

def _strip_thinking(response: str) -> str:
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()


def parse_edits(response: str) -> list[tuple[str, str]]:
    """Extract (search, replace) pairs from the model response, deduplicated."""
    response = _strip_thinking(response)
    response = re.sub(r"```\w*\r?\n?", "", response)
    response = response.replace("\r\n", "\n")
    pattern = (
        r"<{3,7}\s*SEARCH[ \t]*\n"
        r"(.*?)\n"
        r"={3,7}[ \t]*\n"
        r"(.*?)"
        r">{3,7}\s*REPLACE"
    )
    matches = re.findall(pattern, response, re.DOTALL)
    seen = set()
    unique = []
    for search, replace in matches:
        replace = replace.rstrip("\n")
        key = (search, replace)
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


# ---------------------------------------------------------------------------
# Edit application (from rl_pipeline/env.py:164-186)
# ---------------------------------------------------------------------------

def _strip_common_leading_whitespace(text: str) -> str:
    lines = text.splitlines()
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return text
    min_indent = min(len(l) - len(l.lstrip()) for l in non_empty)
    if min_indent == 0:
        return text
    return "\n".join(l[min_indent:] if len(l) >= min_indent else l for l in lines)


def apply_edits(file_path: str, edits: list[tuple[str, str]]) -> bool:
    """Apply SEARCH/REPLACE edits to a file. Returns True if all applied."""
    content = Path(file_path).read_text()
    for search, replace in edits:
        if search in content:
            content = content.replace(search, replace, 1)
        else:
            stripped_search = _strip_common_leading_whitespace(search)
            stripped_replace = _strip_common_leading_whitespace(replace)
            if stripped_search in content:
                content = content.replace(stripped_search, stripped_replace, 1)
            else:
                return False
    Path(file_path).write_text(content)
    return True


# ---------------------------------------------------------------------------
# Training execution (from rl_pipeline/env.py:193-224)
# ---------------------------------------------------------------------------

def run_training(repo_path: str, gpu_id: int = 7, timeout: int = 420) -> dict:
    """Run train.py and parse val_bpb from output."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    result = subprocess.run(
        ["uv", "run", "train.py"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )

    output = result.stdout + result.stderr

    if result.returncode != 0:
        return {"crashed": True, "error": output[-2000:], "output": output[-2000:]}

    parsed: dict = {}
    for line in output.splitlines():
        if line.startswith("val_bpb:"):
            parsed["val_bpb"] = float(line.split(":")[1].strip())
        elif line.startswith("peak_vram_mb:"):
            parsed["peak_vram_mb"] = float(line.split(":")[1].strip())

    if "val_bpb" not in parsed:
        return {"crashed": True, "error": "val_bpb not found in output", "output": output[-2000:]}

    parsed["crashed"] = False
    parsed["output"] = output[-2000:]
    return parsed


# ---------------------------------------------------------------------------
# Git reset (from rl_pipeline/env.py:248-256)
# ---------------------------------------------------------------------------

def git_reset(repo_path: str):
    """Reset train.py to last commit."""
    result = subprocess.run(
        ["git", "checkout", "train.py"],
        cwd=repo_path, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git_reset failed: {result.stderr}")


# ---------------------------------------------------------------------------
# Episode evaluation
# ---------------------------------------------------------------------------

def evaluate_episode(
    repo_path: str,
    parent: State,
    response_text: str,
    gpu_id: int = 7,
    step: int = 0,
) -> dict:
    """
    Parse edits from response, apply to train.py, run training, reset.
    Returns dict with keys: success, child_state, val_bpb, reward, output.
    """
    train_py_path = os.path.join(repo_path, "train.py")

    # Parse edits
    edits = parse_edits(response_text)
    if not edits:
        return {
            "success": False,
            "child_state": None,
            "val_bpb": None,
            "reward": -1.0,
            "output": "No edits found in response",
        }

    # Read original code
    original_code = Path(train_py_path).read_text()

    # Apply edits
    applied = apply_edits(train_py_path, edits)
    if not applied:
        git_reset(repo_path)
        return {
            "success": False,
            "child_state": None,
            "val_bpb": None,
            "reward": -1.0,
            "output": "SEARCH block not found in train.py",
        }

    # Save edited code before running
    edited_code = Path(train_py_path).read_text()

    # Run training
    try:
        result = run_training(repo_path, gpu_id=gpu_id)
    except subprocess.TimeoutExpired:
        result = {"crashed": True, "error": "train.py timed out", "output": "timeout"}

    # Always reset
    git_reset(repo_path)

    crashed = result.get("crashed", False)
    val_bpb = result.get("val_bpb")
    output_text = result.get("output", result.get("error", ""))

    if crashed:
        return {
            "success": False,
            "child_state": None,
            "val_bpb": None,
            "reward": -1.0,
            "output": output_text,
        }

    # Create child state (value = -val_bpb so higher = better)
    child = State(
        timestep=step,
        code=edited_code,
        value=-val_bpb,
        observation=output_text,
    )

    return {
        "success": True,
        "child_state": child,
        "val_bpb": val_bpb,
        "reward": -val_bpb,  # higher = better
        "output": output_text,
    }
