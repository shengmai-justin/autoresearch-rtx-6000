# Plan: TTT-Discover for Autoresearch

## Context

Reimplement TTT-Discover's RL approach in `ttt_autoresearch/`, replacing `tinker` (proprietary) with local HuggingFace + PEFT/LoRA. Task: edit train.py to minimize val_bpb.

**Hardware**: 2x RTX PRO 6000 (96GB each). GPU 6 = LLM inference, GPU 7 = train.py evaluation.
**Mode**: Full RL — PUCT tree search + entropic advantages + importance sampling + KL penalty + LoRA updates.

## Status: IMPLEMENTED

| File | Lines | Purpose |
|---|---|---|
| `ttt_autoresearch/puct.py` | 319 | State + PUCTSampler + JSON persistence |
| `ttt_autoresearch/env.py` | 254 | Prompt building, edit parsing, train.py execution |
| `ttt_autoresearch/model.py` | 173 | HuggingFace model + LoRA, generation with logprobs |
| `ttt_autoresearch/train.py` | 322 | Config + main loop with overlap support |
| `ttt_autoresearch/smoke_test.py` | ~170 | End-to-end pipeline test (mocked model/training) |
| `ttt_autoresearch/pyproject.toml` | 17 | uv-managed venv (torch, transformers, peft) |
| `ttt_autoresearch/setup_and_run.sh` | 18 | Download model + install deps + run |

## Architecture

```
for step in range(NUM_STEPS):
    parent = sampler.sample_state()              # PUCT picks best node

    for g in range(GROUP_SIZE):                  # 4 rollouts per parent
        text, ids, old_lp = generate(prompt)     # GPU 6: HF generate w/ logprobs
        write parent.code → train.py             # edits target parent's code
        apply SEARCH/REPLACE edits
        result = run_training(repo, gpu=7)       # GPU 7: subprocess uv run train.py
        git_reset()                              # always reset

    # RL update
    advantages = entropic_adaptive_beta(rewards) # LOO weights, adaptive beta
    for each episode:
        new_lp = forward_pass(ids)               # with gradient
        ratio = exp(new_lp - old_lp)             # importance sampling
        loss = -(ratio * advantage).mean()
        loss += KL_PENALTY * (new_lp - base_lp)  # KL against frozen base
        loss.backward()
    optimizer.step()
    sampler.save(step)
```

## Key Design Decisions

- **No tinker/chz** — pure PyTorch + PEFT + HuggingFace
- **No flash-attn** — RTX PRO 6000 is Blackwell (SM 120), uses PyTorch SDPA instead
- **Ray parallel eval** — `EVAL_GPUS` list configurable, each eval worker gets its own repo copy, round-robin assignment
- **Overlap gen/eval** — configurable `OVERLAP_GEN_EVAL=True` pipelines generation with evaluation on separate GPUs
- **Parent code written to train.py** before applying edits (edits are generated against parent's code, not committed code)
- **GPU memory** — logit tensors freed immediately after extracting logprobs, full_ids moved to CPU

## Bug Fixes Applied

1. `evaluate_episode` now writes `parent.code` to train.py before applying edits (SEARCH blocks match parent code, not committed code)
2. `generate_with_logprobs` frees GPU logit tensors (~19GB) immediately after extraction
3. `parent.value` check uses `is not None` instead of truthiness

## Future Work

- **Abstract State per task** — Currently `State` is concrete and hardcoded for autoresearch (code = train.py, value = -val_bpb). The original TTT-Discover uses an abstract `State` base class where each task defines its own subclass with a task-specific `construction` field (e.g., number sequences for math, kernel code for CUDA). To support other tasks, make State abstract again with `env_type.create_initial_state()` factory pattern and per-task construction/validation.
- **Multi-replica LLM inference** — Load model on multiple GPUs for parallel generation. Requires LoRA weight sync across replicas after each RL update.
- **Checkpoint resume** — Save/load LoRA weights + optimizer state to resume training from a specific step.

## Timing

Per step (sequential): 4 generations (~8 min) + 4 evals (~28 min) + RL update (~1 min) ≈ **37 min/step**
Per step (overlapped): 1 gen + 3 overlapped gen+eval + 1 eval + RL ≈ **31 min/step**
50 steps ≈ **26-31 hours**

## Verification (on server)

```bash
cd ttt_autoresearch

# 1. Smoke test (no GPU needed, uses tiny-gpt2)
uv run python smoke_test.py

# 2. Full run
./setup_and_run.sh

# 3. Or manually:
uv sync
uv run python train.py
```
