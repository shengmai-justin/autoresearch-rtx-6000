# Plan: Autoresearch with PUCT Tree Search (Simplified)

## Context

We want PUCT tree search to guide an LLM in optimizing train.py (minimizing val_bpb). The original plan used TTT-Discover's full framework (shims, async rollouts, Environment subclasses, etc.) but this was **over-engineered** — we needed ~450 lines of useful code (PUCTSampler + State) but wrote ~900 lines of adapters and shims to use it.

**Rewrite**: Extract just PUCTSampler and State into a standalone file. Write a simple synchronous loop. No shims, no async, no ttt_discover imports.

**Hardware**: 2x RTX PRO 6000 (96GB each). GPU 6 = LLM inference, GPU 7 = train.py evaluation.
**Mode**: Inference-only. No RL weight updates.

## Status: IMPLEMENTED

Two files created, old files deleted:

| File | Lines | Purpose |
|---|---|---|
| `discover/puct.py` | 364 | Self-contained State + PUCTSampler + JSON persistence. Only depends on numpy. |
| `discover/run_autoresearch.py` | 332 | Prompt building, edit parsing, training runner, model loading, main loop. |

Old files removed:
- `ttt_discover/local_backend/` (shims.py, completer.py, __init__.py)
- `ttt_discover/environments/autoresearch/` (env.py, reward.py, __init__.py)
- `pyproject.toml` restored to original

## Architecture

```
for step in range(num_steps):
    parent = sampler.sample_state()          # PUCT picks best node
    prompt = build_prompt(repo_path, parent) # train.py + state context
    response = generate(model, tokenizer, prompt)  # HF model.generate()
    edits = parse_edits(response)            # regex SEARCH/REPLACE
    apply_edits(train_py, edits)             # modify file
    result = run_training(repo_path, gpu)    # subprocess uv run train.py
    git_reset(repo_path)                     # always reset
    child = State(value=-val_bpb, ...)       # higher = better for PUCT
    sampler.update_state(child, parent)      # add to tree
    sampler.save(step)                       # persist to disk
```

No async. No shims. No subclasses. One state at a time — no race condition on train.py.

## What Was Kept from TTT-Discover

Extracted into `puct.py` (self-contained, no ttt_discover imports):
- `State` class — concrete (not ABC), with `to_dict`, `from_dict`, `to_prompt`
- `PUCTSampler` — full PUCT algorithm with visit counts, rank-based prior, exploration bonus
- JSON persistence — atomic writes, file locking, step-based checkpoints

## What Was Kept from rl_pipeline/env.py

Copied into `run_autoresearch.py`:
- `parse_edits()` — regex SEARCH/REPLACE extraction with dedup
- `apply_edits()` — exact match + whitespace-stripped fallback
- `run_training()` — subprocess with CUDA_VISIBLE_DEVICES, output parsing
- `git_reset()` — revert train.py to last commit
- `SYSTEM_PROMPT`, `EDIT_FORMAT` — prompt constants

## What Was Removed (vs old plan)

- **No tinker/chz shims** — we don't import ttt_discover at all
- **No importlib bootstrap** — no import chain to work around
- **No Environment/BaseRewardEvaluator subclasses** — just functions
- **No DatasetConfig/SingleProblemDataset/ProblemGroupBuilder** — just a for-loop
- **No async/await or ThreadPoolExecutor** — synchronous
- **No Qwen3Renderer** — using `tokenizer.apply_chat_template()` directly
- **No pyproject.toml changes** — ttt_discover is untouched

## Verification (run on server only)

```bash
cd discover

# 1. Dry run — verify setup without GPU
python run_autoresearch.py --dry-run

# 2. Full run
python run_autoresearch.py

# 3. Full run with custom settings
python run_autoresearch.py --model-name ./models/Qwen3.5-27B --baseline-bpb 0.95 --num-steps 50
```
