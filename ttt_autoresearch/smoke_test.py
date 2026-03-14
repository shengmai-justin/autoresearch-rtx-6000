"""
Smoke test: verify the full pipeline end-to-end with mocked model/training.
Run on server: cd ttt_autoresearch && uv run python smoke_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import shutil

import torch
import numpy as np


def test_state_and_puct():
    """Test State creation, serialization, and PUCTSampler logic."""
    print("1. Testing State + PUCTSampler...", end=" ")
    from puct import State, PUCTSampler

    # Create state
    s0 = State(timestep=0, code="print('hello')", value=-0.95, observation="val_bpb: 0.95")
    assert s0.value == -0.95
    assert s0.code == "print('hello')"

    # Serialization round-trip
    d = s0.to_dict()
    s0_copy = State.from_dict(d)
    assert s0_copy.id == s0.id
    assert s0_copy.value == s0.value

    # to_prompt
    prompt = s0.to_prompt()
    assert "0.950000" in prompt

    # PUCTSampler
    with tempfile.TemporaryDirectory() as tmpdir:
        sampler = PUCTSampler(initial_state=s0, log_dir=tmpdir, puct_c=1.0)
        assert sampler.buffer_size() == 1

        # Sample
        parent = sampler.sample_state()
        assert parent.id == s0.id

        # Add child
        child = State(timestep=1, code="print('better')", value=-0.90, observation="val_bpb: 0.90")
        sampler.update_state(child, parent)
        assert sampler.buffer_size() == 2

        # Failed rollout
        sampler.record_failed_rollout(parent)

        # Save/load round-trip
        sampler.save(0)
        sampler2 = PUCTSampler(initial_state=s0, log_dir=tmpdir, resume_step=0)
        assert sampler2.buffer_size() == 2

        # Best state
        best = sampler.best_state()
        assert best.value == -0.90

    print("OK")


def test_env_parsing():
    """Test edit parsing and application."""
    print("2. Testing env (parse_edits, apply_edits)...", end=" ")
    from env import parse_edits, apply_edits, build_prompt
    from puct import State

    # parse_edits
    response = """
<think>Let me think about this...</think>

I'll change the learning rate.

<<<<<<< SEARCH
lr = 0.001
=======
lr = 0.0005
>>>>>>> REPLACE
"""
    edits = parse_edits(response)
    assert len(edits) == 1
    assert edits[0] == ("lr = 0.001", "lr = 0.0005")

    # No edits
    assert parse_edits("no edits here") == []

    # apply_edits
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("lr = 0.001\nbatch_size = 32\n")
        f.flush()
        path = f.name

    try:
        ok = apply_edits(path, edits)
        assert ok
        with open(path) as f:
            content = f.read()
        assert "lr = 0.0005" in content
        assert "batch_size = 32" in content
    finally:
        os.unlink(path)

    # build_prompt
    state = State(timestep=0, code="x = 1", value=-0.95)
    prompt = build_prompt(state)
    assert "val_bpb" in prompt
    assert "SEARCH" in prompt

    print("OK")


def test_entropic_advantages():
    """Test entropic adaptive beta advantage computation."""
    print("3. Testing entropic advantages...", end=" ")
    sys.path.insert(0, ".")
    from train import compute_entropic_advantages

    # Uniform rewards → advantages near 0
    adv = compute_entropic_advantages([1.0, 1.0, 1.0, 1.0])
    assert torch.allclose(adv, torch.zeros(4), atol=1e-5), f"Expected ~0, got {adv}"

    # One high reward → that one gets positive advantage
    adv = compute_entropic_advantages([-1.0, -1.0, -1.0, 0.5])
    assert adv[3] > 0, f"Best rollout should have positive advantage, got {adv}"
    assert adv[0] < 0, f"Worst rollout should have negative advantage, got {adv}"

    # Single reward → zeros
    adv = compute_entropic_advantages([0.5])
    assert torch.allclose(adv, torch.zeros(1))

    print("OK")


def test_model_logprobs():
    """Test compute_response_logprobs / compute_base_logprobs shapes (tiny model)."""
    print("4. Testing model logprob functions (CPU, tiny)...", end=" ")
    from model import compute_response_logprobs, compute_base_logprobs
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig, TaskType

    model_name = "sshleifer/tiny-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(model_name)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4, lora_alpha=8, lora_dropout=0.0,
        target_modules=["c_attn"],
    )
    model = get_peft_model(base_model, lora_config)

    # Fake input: 5 prompt tokens + 3 response tokens
    full_ids = torch.tensor([50256, 1, 2, 3, 4, 5, 6, 7])
    prompt_len = 5

    # compute_response_logprobs (with grad)
    lp = compute_response_logprobs(model, tokenizer, full_ids, prompt_len, temperature=1.0)
    assert lp.shape == (3,), f"Expected shape (3,), got {lp.shape}"
    assert lp.requires_grad

    # compute_base_logprobs (no grad)
    base_lp = compute_base_logprobs(model, tokenizer, full_ids, prompt_len, temperature=1.0)
    assert base_lp.shape == (3,), f"Expected shape (3,), got {base_lp.shape}"
    assert not base_lp.requires_grad

    print("OK")


def test_full_pipeline_mocked():
    """End-to-end pipeline with mocked generation and training."""
    print("5. Testing full pipeline (mocked)...", end=" ")
    from puct import State, PUCTSampler
    from train import compute_entropic_advantages
    from env import build_prompt, parse_edits, apply_edits

    with tempfile.TemporaryDirectory() as tmpdir:
        # Simulate: original code
        original_code = "lr = 0.001\nbatch_size = 32\nepochs = 10\n"

        # Init
        initial_state = State(timestep=0, code=original_code, value=-0.95)
        sampler = PUCTSampler(initial_state=initial_state, log_dir=tmpdir)

        # Simulate 2 steps x 2 rollouts
        for step in range(2):
            parent = sampler.sample_state()
            prompt = build_prompt(parent)
            assert len(prompt) > 0

            rewards = []
            children = []
            for g in range(2):
                # Mock generation response
                fake_response = f"""
<<<<<<< SEARCH
lr = 0.001
=======
lr = {0.0005 - g * 0.0001}
>>>>>>> REPLACE
"""
                edits = parse_edits(fake_response)
                assert len(edits) == 1

                # Mock val_bpb result
                fake_bpb = 0.95 - step * 0.02 - g * 0.01
                child = State(
                    timestep=step,
                    code=original_code.replace("lr = 0.001", f"lr = {0.0005 - g * 0.0001}"),
                    value=-fake_bpb,
                    observation=f"val_bpb: {fake_bpb}",
                )
                rewards.append(-fake_bpb)
                children.append(child)

            # Advantages
            advantages = compute_entropic_advantages(rewards)
            assert advantages.shape == (2,)

            # Update tree
            for child in children:
                sampler.update_state(child, parent)

        sampler.save(1)
        assert sampler.buffer_size() >= 3  # initial + children
        best = sampler.best_state()
        assert best.value > -0.95

    print("OK")


if __name__ == "__main__":
    print("=" * 50)
    print("TTT-Autoresearch Smoke Test")
    print("=" * 50)

    test_state_and_puct()
    test_env_parsing()
    test_entropic_advantages()
    test_model_logprobs()
    test_full_pipeline_mocked()

    print("=" * 50)
    print("All tests passed!")
    print("=" * 50)
