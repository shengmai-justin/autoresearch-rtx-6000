# TTT-Discover Paper Reading Note

This note is meant to help you understand the **paper first**, so that when you open the repo you already know what to look for.

---

## 1. One-paragraph summary

The paper proposes **test-time training for scientific/engineering discovery**. Instead of asking a language model to solve a hard problem in one shot, it lets the model repeatedly:

1. propose a candidate solution,
2. execute or evaluate it in a real environment,
3. get a scalar reward,
4. update the model online,
5. reuse promising previous states/solutions,
6. continue searching.

The main claim is that **discovery problems care about rare excellent solutions, not average-quality samples**, so ordinary expected-reward RL is not the best objective. Their method uses:

- an **entropic utility objective** to emphasize rare high-reward rollouts,
- **adaptive beta** to control how aggressive that emphasis is,
- **PUCT-based reuse** to revisit promising past states,
- **LoRA-based online adaptation** instead of full finetuning.

---

## 2. What problem are they trying to solve?

The paper is about tasks where:

- there is a **real external evaluator**,
- solutions can be **executed / scored**,
- you want the **best solution found**,
- and the reward is often **dense or scalar**.

Examples in the paper include:

- math inequality / bound optimization,
- CUDA or kernel engineering,
- algorithmic competition tasks,
- biology or denoising tasks.

This is different from standard RL post-training on a fixed dataset. Here, the model is being adapted **during the actual search on a specific problem instance**.

---

## 3. The key conceptual shift

### Standard view
A policy tries to maximize **average reward**.

### Their view
For discovery, we usually care about:

> "Can the system find one unusually good solution?"

That is more like **best-of-many search** than plain expected-value optimization.

So the paper tries to optimize a distribution that pays much more attention to **high-reward tails**.

---

## 4. The main algorithm in plain English

For one problem:

1. Start with a pretrained LLM.
2. Keep a history/buffer of previous states and solutions.
3. Choose an initial state from that history using **PUCT-style reuse**.
4. Sample one or more actions from the model.
5. Execute/evaluate them in the environment.
6. Convert outcome to a scalar reward.
7. Add the new trajectory/result back into history.
8. Update the model online using their entropic RL objective.
9. Repeat.

So it is both:

- **search over solutions**, and
- **online policy adaptation**.

---

## 5. What exactly is the RL objective?

Their core objective is the **entropic utility objective**:

\[
J_\beta(\theta; s) = \log \mathbb{E}_{\tau \sim \pi_\theta(\cdot \mid s)} \left[e^{\beta r(\tau; s)}\right]
\]

Where:

- \(s\) is the current initial state / context,
- \(\tau\) is a sampled rollout / candidate,
- \(r(\tau; s)\) is the scalar reward,
- \(\beta\) controls how strongly high-reward samples are emphasized.

### Intuition

- If \(\beta\) is small, this behaves closer to normal expected reward.
- If \(\beta\) is larger, high-reward samples get amplified exponentially.
- In spirit, it moves the system toward **"care a lot about rare excellent attempts"**.

---

## 6. What is the effective "advantage" or training signal?

They derive importance-style weights:

\[
w_\beta(\tau \mid s) = \frac{e^{\beta r(\tau; s)}}{\mathbb{E}_{\pi_\theta}[e^{\beta r(\tau; s)}]}
\]

Then a centered version:

\[
A_\beta(\tau \mid s) = w_\beta(\tau \mid s) - 1
\]

And with regularization, the practical shaped signal becomes roughly:

\[
A(a; s) = w_\beta^{(s)}(a) - 1 - \lambda \log \frac{\pi_\theta(a \mid s)}{\pi_{\theta_0}(a \mid s)}
\]

So in plain words:

- good samples get **more positive weight**,
- bad samples get less or negative relative pressure,
- the policy is also penalized for moving too far from the initial model.

This is why the method is clearly **RL**, but **not just standard PPO/GRPO**.

---

## 7. What is adaptive beta?

They do **not** use one fixed beta for all states.

Instead, they define a tilted distribution and choose beta so that the KL shift from the original sampling distribution stays within a budget.

### Why this matters

If beta is too small:
- the policy acts too much like average-reward RL.

If beta is too large:
- updates become unstable and can over-focus on a tiny number of samples.

Adaptive beta is their way to say:

> "Push toward high-reward discoveries, but only as aggressively as the current reward landscape safely allows."

---

## 8. What is PUCT reuse doing?

This is a major part of the paper.

They do **not** always restart from scratch.
They keep a history of prior states/solutions and use a **PUCT-style selection rule** to decide where to branch from next.

### Intuition

This gives the method an MCTS-like flavor:

- revisit promising partial solutions,
- avoid wasting all budget on fresh random starts,
- balance exploitation of good states with exploration of less-tried ones.

### Why it matters

Their ablations suggest that:

- **expected reward + no reuse** is weak,
- **entropic objective + PUCT reuse** is much stronger.

So the gains are not just from “doing RL”; they come from the combination of:

- better objective,
- better state reuse.

---

## 9. What is the reward function?

The paper **does have rewards**, but there is **no single universal reward formula**.

The framework only requires a scalar reward \(R(s,a)\).
Each benchmark plugs in its own evaluator.

### Examples from the paper

- **Math minimization / upper bound tasks**: reward based on **inverse bound**.
- **Math maximization tasks**: reward can be proportional to the achieved bound/value.
- **Kernel engineering**: reward based on **inverse runtime**.
- **Algorithmic tasks**: reward based on **test score**.
- **Biology / denoising**: reward based on **inverse MSE** or MSE-derived score.
- Invalid outputs, crashes, or constraint violations often get **0 reward**.

### Important takeaway

The method is not limited to one domain. It works whenever you can define:

- a candidate solution,
- an evaluator,
- and a scalar score.

---

## 10. Are they full-finetuning the model?

No. The paper uses **LoRA-based adaptation**, not full-model finetuning.

### What that means

- Base model weights stay frozen.
- LoRA adapters are trained online.
- This makes the test-time RL updates much cheaper and more practical.

The paper reports **LoRA rank 32**.

### Shape intuition
If a linear layer is:

\[
W \in \mathbb{R}^{d_{out} \times d_{in}}
\]

LoRA adds:

\[
A \in \mathbb{R}^{32 \times d_{in}}, \quad B \in \mathbb{R}^{d_{out} \times 32}
\]

So the update is low-rank and much smaller than updating the whole matrix.

---

## 11. Which models do they use?

### Main model
- **gpt-oss-120b** is the main model used for the headline system.

### Comparison model
- **Qwen3-8B** is used mainly for a fairer comparison to **ThetaEvolve-style Qwen-based math experiments**.

### Important interpretation
The paper is **not** saying 8B is their main best system. The stronger main setup uses the larger model.

The 8B result matters because it shows the method can still work with a smaller model on structured discovery problems.

---

## 12. Why can 8B still work at all?

Because the paper does not rely on one-shot reasoning.

An 8B model can still do well if it gets:

- iterative search,
- real scalar feedback,
- online adaptation,
- reuse of promising states,
- structured optimization problems.

So the success of the 8B system should be understood as:

> the **loop + evaluator + reuse + RL objective** make the base model much stronger than plain one-shot generation.

It does **not** mean an 8B model is universally enough for open-ended autonomous research.

---

## 13. Training setup you should remember

The paper reports a setup along these lines:

- **50 training steps**,
- **512 rollouts per step**,
- **8 groups of 64 rollouts**,
- **LoRA rank 32**,
- **Adam** optimizer for adaptation,
- typically **1 gradient step per batch**,
- **importance sampling correction** for sampler/learner mismatch.

This is the practical recipe behind the method.

---

## 14. What are the paper's real contributions?

If you strip away the wording, the paper contributes three main things:

### (A) A better objective for discovery
They argue average reward is misaligned with “find one excellent solution.”

### (B) A better search / reuse mechanism
PUCT-style reuse lets the system branch from good prior states instead of always restarting.

### (C) A practical online adaptation recipe
They show this can be done with open-weight LLMs using **LoRA**, not just full giant retraining.

---

## 15. What should you look for in the repo?

When you open the repo, try to map files/functions into these paper concepts:

### A. Environment / evaluator
Look for code that:
- executes a candidate,
- checks correctness or constraints,
- measures runtime / score / error,
- returns a scalar reward.

### B. Sampling / rollout generation
Look for code that:
- prompts the model,
- samples code / text / actions,
- groups multiple rollouts,
- records logprobs if needed.

### C. History / replay / reuse buffer
Look for code that:
- stores prior states,
- stores actions and rewards,
- picks a prior state to branch from,
- tracks visitation counts or scores.

### D. PUCT selection
Look for code that:
- computes some score combining reward and exploration bonus,
- chooses which previous state/node to expand next.

### E. RL update
Look for code that:
- computes reward weights,
- computes beta or adaptive beta,
- forms an advantage-like signal,
- applies KL regularization,
- updates LoRA parameters.

### F. Logging / metrics
Look for code that records:
- best-so-far solution,
- reward distributions,
- beta / KL / training stats,
- success rate,
- comparison against ablations.

---

## 16. A useful mental model before reading code

You can think of the full system as:

\[
\text{LLM policy} + \text{search tree / history} + \text{real evaluator} + \text{online RL update}
\]

It is **not** just:

- plain sampling,
- plain beam search,
- plain PPO,
- plain evolutionary search,
- plain MCTS.

It borrows ideas from several of these.

---

## 17. How this differs from Karpathy's `autoresearch`

If you are also thinking about Karpathy's repo, the difference is:

### Karpathy `autoresearch`
- agent edits `train.py`,
- run a 5-minute training experiment,
- check `val_bpb`,
- keep or discard,
- repeat.

### TTT-Discover
- formal online RL objective,
- explicit reward shaping,
- adaptive beta,
- PUCT reuse over states,
- LoRA adaptation of the model itself.

So Karpathy's repo is closer to an **agentic experiment loop**, while this paper is a more explicit **test-time RL + reuse** method.

---

## 18. Questions to ask yourself while reading the repo

Use these questions as a checklist:

1. Where is the **reward** computed?
2. What exactly is a **state** in this implementation?
3. What exactly is an **action**?
4. What counts as a **trajectory**?
5. How are **invalid outputs** handled?
6. Where is the **history / buffer** stored?
7. How is **PUCT** implemented?
8. Where is **adaptive beta** computed?
9. Where is the **KL regularization** applied?
10. Which parameters are actually trainable — **LoRA only** or more?
11. How do they decide which candidate becomes the **best-so-far** solution?
12. How do they compare against **naive expected-reward RL** or **no-reuse baselines**?

---

## 19. What you should not misunderstand

### Misunderstanding 1
**"They don't use RL."**

False. They do use RL, but not plain off-the-shelf PPO/GRPO as the main idea.

### Misunderstanding 2
**"They don't have a reward."**

False. They do have rewards; the reward is task-specific.

### Misunderstanding 3
**"They full-finetune the whole model online."**

False. They use **LoRA-based adaptation**.

### Misunderstanding 4
**"Qwen3-8B is their main flagship setup."**

Not really. It is mainly used as an additional comparison point.

### Misunderstanding 5
**"This only works for continuous-control style rewards."**

Too narrow. The method needs a scalar reward. That reward can come from runtime, score, inverse error, and in related contexts can even be binary.

---

## 20. Minimal pseudo-code view

```text
initialize pretrained policy pi_theta
initialize history H

repeat:
    choose initial state s from H using PUCT-style reuse
    sample candidate actions / trajectories from pi_theta(. | s)
    evaluate each candidate in the real environment
    compute scalar reward r
    add results back into H
    choose adaptive beta for current state / batch
    compute entropic weights exp(beta * r)
    form advantage-like weighted signal + KL regularization
    update LoRA parameters of pi_theta
end
```

---

## 21. What matters most if you want to reproduce or extend it

If you want to build on this paper, the most important pieces are probably:

1. **A good evaluator / reward**
2. **A stable reuse strategy**
3. **A strong enough base model**
4. **Careful online adaptation via LoRA**
5. **Reward normalization / beta control**

My view: the biggest danger is thinking the magic is only in "RL". It is more accurate to say the method works because it couples:

- good search,
- real execution feedback,
- and an objective aimed at rare high-value discoveries.

---

## 22. Final takeaway

If you only remember one sentence, remember this:

> TTT-Discover is a **test-time RL-for-discovery** system that updates a frozen-base LLM through **LoRA**, scores candidates with **real task rewards**, emphasizes rare high-reward attempts through an **entropic objective**, and revisits promising partial solutions using **PUCT-style reuse**.

---

## 23. Repo-reading companion checklist

When you open the repo, try to annotate files with these labels:

- `environment/evaluator`
- `sampling/generation`
- `history/buffer`
- `PUCT/reuse`
- `reward shaping`
- `beta / KL logic`
- `LoRA / trainable params`
- `optimizer / update step`
- `logging / best solution`

If you can identify those 9 pieces, you will understand most of the implementation.
