# RL Training Workflow in the Paper

Here is the **full RL training workflow** in the paper, in the order it actually happens.

## Big picture

Their method is **TTT-Discover**: a combination of

- **test-time training** of the policy with an **entropic RL objective**, and
- **PUCT-based reuse** to choose which previously found state/solution to continue from next. 

So it is not just “sample a lot and pick the best,” and it is not plain PPO/GRPO either. The loop is: **reuse a promising prior state → sample many rollouts from the current model → evaluate with real reward → do one RL update → repeat**. 

## Step 0: Initialize the model and history

They start from a base model, mainly **gpt-oss-120b**, and attach **LoRA rank 32** adapters. The model is then updated for **50 training steps** using **Adam** with learning rate \(4\times 10^{-5}\). The KL penalty coefficient is usually **0.1**, and **0.01** for algorithm engineering. 

You can think of the history buffer \(H\) as storing previously attempted states/solutions and their rewards. That buffer is what the reuse policy looks at before the next rollout batch. 

## Step 1: Choose an initial state from the reuse buffer

At each training iteration, they do **not always start from scratch**. Instead, they select an initial state using a **PUCT-inspired rule**. The score for a state combines:

- \(Q(s)\): the **best reward** ever achieved from that state’s descendants,
- \(P(s)\): a prior favoring states that already rank highly by reward,
- an exploration bonus using visit counts \(n(s)\) and total expansions \(T\). 

A key design choice is that \(Q(s)\) uses the **maximum child reward**, not the mean. Their reasoning is: discovery problems care about whether a state can lead to a **great solution**, not whether it is good on average. 

So the reuse step is basically:  
**pick a promising previous solution branch that is worth extending further.** 

## Step 2: Sample rollouts from the current policy

Once an initial state is chosen, they generate rollouts from the **current adapted policy**, not the untouched original model. Each training step uses a batch of **512 rollouts**, arranged as **8 groups of 64**. Within each group, all 64 rollouts use the **same selected initial state and context** from the reuse buffer. Different groups can use different reused states. 

Their sampling setup is:

- context window: **32768 tokens**
- temperature: **1.0**
- prompt length + thinking-token limit: **26000**
- reasoning effort: **high**. 

A rollout ends when the context window is exhausted or the model emits EOS. They also force the model to leave enough space for the final response, such as long algorithm code. 

## Step 3: Evaluate each rollout with a task reward

Each sampled rollout is executed or checked in the task environment and assigned a **scalar reward**. The exact reward depends on the domain, but the method always needs a number that says how good that rollout was. Examples in the paper include runtime-based rewards, score-based rewards, and inverse-error rewards. Invalid or constraint-violating outputs can receive **0** reward. 

So after this step, they have a batch of 512 trajectories/actions with associated rewards. 

## Step 4: Convert rewards into entropic weights

Instead of optimizing ordinary expected reward, they optimize the **entropic utility objective**:

\[
J_\beta(\theta;s)=\log \mathbb{E}_{\tau\sim\pi_\theta(\cdot|s)}\left[e^{\beta r(\tau;s)}\right].
\]

This objective upweights high-reward rollouts **exponentially**. The resulting policy-gradient weight is

\[
w_\beta(\tau|s)=\frac{e^{\beta r(\tau;s)}}{\mathbb{E}_{\pi_\theta}[e^{\beta r(\tau;s)}]},
\]

and the mean-baselined advantage is

\[
A_\beta(\tau|s)=w_\beta(\tau|s)-1.
\]

So compared with ordinary RL, a rollout that is only slightly better gets some extra weight, while a rare very strong rollout can dominate the update much more. That is exactly what they want for discovery problems. 

## Step 5: Adapt \(\beta\) automatically

They found that a fixed \(\beta\) is hard to tune: too large early can destabilize training, too small later makes advantages vanish when improvements become tiny. So they choose \(\beta(s)\) **adaptively per initial state** by defining a tilted distribution \(q_\beta\) and enforcing a **KL budget**:

\[
KL(q_\beta(\cdot|s)\,\|\,\pi_\theta(\cdot|s))=\gamma,
\]

with \(\gamma=\ln 2\). 

Intuitively, this means:  
**push hard toward better rollouts, but not so hard that the effective update jumps too far from the current policy.** 

## Step 6: Add KL regularization to the base policy

Their shaped advantage also includes a KL penalty against the **initial policy**:

\[
A(a;s)=w_\beta^{(s)}(a)-1-\lambda \log \frac{\pi_\theta(a|s)}{\pi_{\theta_0}(a|s)}.
\]

This regularization discourages the adapted policy from drifting too far away from the original starting model. 

So the training signal is not just “higher reward = push probability up.” It is:

- raise probability for entropically reweighted good rollouts,
- subtract the baseline,
- penalize moving too far from the starting policy. 

## Step 7: Do one gradient update on the batch

After computing the gradients from the 512-rollout batch, they take **1 gradient step** on the whole batch. They explicitly say they do **not** take extra off-policy optimization steps. They also apply **importance-sampling ratio correction** because of sampler/learner mismatch in their RL infrastructure. 

So the update pattern is:

- collect batch,
- compute entropic RL gradient,
- do one optimizer step on LoRA parameters,
- move to the next iteration. 

## Step 8: Add results back into the history and repeat

The new states/solutions and their rewards are added back into the history buffer. On the next iteration, PUCT can choose among these accumulated states to decide where to branch from next. This is why the method can progressively extend a promising line of search instead of always restarting from the original state. 

They repeat this for **50 training steps**. The paper estimates that, with average prompt and sampling lengths, a full run of 50 steps × 512 rollouts costs around **$500 on Tinker**. 

## Why this workflow matters

Their ablations show that each major piece matters:

- replacing the entropic objective with ordinary expected reward hurts,
- removing test-time training hurts,
- replacing PUCT reuse with no reuse hurts a lot,
- “naive test-time RL” with expected reward and no reuse performs very poorly. 

So the method’s strength is the **combination**:

1. **reuse promising earlier states**,  
2. **sample many rollouts**,  
3. **evaluate with real rewards**,  
4. **train with an entropic objective**,  
5. **regularize with KL**,  
6. **repeat online**. 

## One-sentence summary

A precise summary is:

> They do **online test-time RL on LoRA adapters** by repeatedly choosing a promising prior state via **PUCT**, generating **512 rollouts** from the current model, scoring them with a real task reward, converting rewards into **adaptive entropic policy-gradient weights**, taking **one KL-regularized gradient step**, and then feeding the new results back into the reuse buffer for the next round. 
