"""
All settings in one place. Edit this file to change any knob.
No YAML, no CLI args for now — just change the values and run.
"""

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_NAME = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"  # HuggingFace model ID
MODEL_DIR = "./models/DeepSeek-R1-0528-Qwen3-8B"       # local path after download
VLLM_PORT = 8000                                        # vLLM server port
VLLM_GPU = 6                                            # GPU index for LLM

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
REPO_PATH = ".."                                        # path to the autoresearch repo (parent dir)
TRAIN_GPU = 7                                           # GPU index for train.py
EVAL_TIME_BUDGET = 300                                  # seconds per train.py run (300 = full, 60 = proxy)

# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------
MAX_NEW_TOKENS = 8192           # max tokens the model can generate per response
TEMPERATURE = 1.0               # sampling temperature during episode collection
BASELINE_BPB = 1.0              # initial val_bpb (updated after first baseline run)
CRASH_PENALTY = -1.0            # reward for crashed runs
KEEP_IF_IMPROVED = False        # True = keep good changes (sequential autoresearch mode)
                                # False = always reset (independent episodes for RL)

# ---------------------------------------------------------------------------
# RL Training
# ---------------------------------------------------------------------------
ALGO = "grpo"                   # "grpo", "ppo", or "none" (ablation)
EPISODES_PER_STEP = 4           # episodes collected before each RL update
NUM_STEPS = 100                 # total RL training steps
LEARNING_RATE = 1e-6            # RL learning rate
CHECKPOINT_DIR = "./checkpoints"
CHECKPOINT_INTERVAL = 10        # save every N steps
TRAJECTORY_DIR = "./trajectories"

# ---------------------------------------------------------------------------
# GRPO-specific
# ---------------------------------------------------------------------------
GRPO_CLIP_LOW = 0.2
GRPO_CLIP_HIGH = 0.28
GRPO_KL_COEFF = 0.0            # 0 = no KL penalty

# ---------------------------------------------------------------------------
# PPO-specific (for future use)
# ---------------------------------------------------------------------------
PPO_CLIP = 0.2
PPO_VALUE_COEFF = 0.5
PPO_EPOCHS = 4
