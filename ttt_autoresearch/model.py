"""
Model loading, LoRA, generation with logprobs, training utilities.

Replaces tinker with local HuggingFace + PEFT.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_dir: str,
    gpu_id: int = 6,
    lora_rank: int = 32,
    lora_alpha: int = 64,
) -> tuple:
    """Load base model + LoRA adapter. Returns (model, tokenizer)."""
    device = f"cuda:{gpu_id}"

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2",
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation with per-token logprobs
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_with_logprobs(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 32768,
    temperature: float = 1.0,
) -> tuple[str, torch.Tensor, torch.Tensor, int]:
    """
    Generate a response and collect per-token logprobs.

    Returns:
        (text, token_ids, logprobs, prompt_len)
        - text: decoded response string
        - token_ids: full sequence (prompt + response) as 1D tensor
        - logprobs: per-token logprobs for response tokens only (1D tensor, len = num_new_tokens)
        - prompt_len: number of prompt tokens
    """
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        return_dict_in_generate=True,
        output_logits=True,
    )

    full_ids = outputs.sequences[0]  # [prompt_len + new_tokens]
    new_ids = full_ids[prompt_len:]

    # Extract logprobs from logits
    # outputs.logits is a tuple of (num_new_tokens,) tensors, each [1, vocab]
    logprobs_list = []
    for t, logits_t in enumerate(outputs.logits):
        if temperature > 0:
            log_probs = torch.log_softmax(logits_t[0] / temperature, dim=-1)
        else:
            log_probs = torch.log_softmax(logits_t[0], dim=-1)
        token_id = new_ids[t]
        logprobs_list.append(log_probs[token_id].item())

    logprobs = torch.tensor(logprobs_list, dtype=torch.float32)
    text = tokenizer.decode(new_ids, skip_special_tokens=True)

    return text, full_ids, logprobs, prompt_len


# ---------------------------------------------------------------------------
# Compute response logprobs (with gradient, for training)
# ---------------------------------------------------------------------------

def compute_response_logprobs(
    model,
    tokenizer,
    full_ids: torch.Tensor,
    prompt_len: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Forward pass to get per-token logprobs for response tokens.
    Returns tensor with gradient attached (for backprop).
    """
    input_ids = full_ids.unsqueeze(0).to(model.device)  # [1, seq_len]
    outputs = model(input_ids=input_ids)
    logits = outputs.logits[0]  # [seq_len, vocab]

    # Response tokens start at prompt_len
    # logits[t] predicts token at position t+1
    response_logits = logits[prompt_len - 1 : -1]  # [num_response, vocab]
    response_ids = full_ids[prompt_len:]             # [num_response]

    if temperature > 0:
        log_probs = torch.log_softmax(response_logits / temperature, dim=-1)
    else:
        log_probs = torch.log_softmax(response_logits, dim=-1)

    token_logprobs = log_probs.gather(1, response_ids.unsqueeze(1).to(model.device)).squeeze(1)
    return token_logprobs  # [num_response], has grad


# ---------------------------------------------------------------------------
# Compute base model logprobs (no gradient, adapter disabled)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_base_logprobs(
    model,
    tokenizer,
    full_ids: torch.Tensor,
    prompt_len: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Forward pass with LoRA adapters disabled to get base model logprobs.
    Returns detached tensor (no gradient).
    """
    model.disable_adapter_layers()
    try:
        input_ids = full_ids.unsqueeze(0).to(model.device)
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[0]

        response_logits = logits[prompt_len - 1 : -1]
        response_ids = full_ids[prompt_len:]

        if temperature > 0:
            log_probs = torch.log_softmax(response_logits / temperature, dim=-1)
        else:
            log_probs = torch.log_softmax(response_logits, dim=-1)

        token_logprobs = log_probs.gather(1, response_ids.unsqueeze(1).to(model.device)).squeeze(1)
        return token_logprobs
    finally:
        model.enable_adapter_layers()
