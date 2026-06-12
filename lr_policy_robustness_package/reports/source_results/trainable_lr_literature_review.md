# Trainable Learning Rate — Literature Review and Repo-Specific Proposals

Date: 2026-05-12

This note maps recent (2023–2025) work on learnable / adaptive learning-rate
policies to the failure modes documented in `results/final_resume.md` and
`results/2h_curve/RESULTS.md`, and proposes concrete changes to the codebase.

---

## 1. What the repo actually does today

Quick recap so the proposals below are not generic:

- Per epoch we compute a **30-dim "loss-landscape probe"** — loss after one SGD
  step on a held-out batch at each LR in
  `LR_GRID = [0.0005 ... 2.5]` (`trainer/policy_data.py` and `research(...)` in
  `trainer/my_research.py`).
- A neural policy
  (`GRULRPolicy` / `ModularLRPolicy` / `AttentionModularLRPolicy` /
  `CurveLRPolicy` in `model/lr_policy.py`) takes a window of
  `[landscape | logit-mean+std | time]` features and predicts:
  - `raw_lr` — direct scalar LR (fails: 60% vs 70% baseline).
  - `cosine_multiplier` — multiplier on top of cosine (matches baseline ~69%).
  - `curve_argmin` — full predicted landscape, argmin → LR (catastrophic 32%;
    online MPC reduces curve MSE by 88% but the argmin saturates at the highest
    LR late in training).
- Offline pretraining on `features_v3_*.npy` plus optional online MPC updates
  with lookahead-2 (`deploy_curve_policy_step`).

Diagnosis (already correct in `final_resume.md`): **late in training the
gradient norm shrinks, the 1-step probe flattens, and the predicted argmin
drifts to LR=2.5**, which destabilizes the student.

---

## 2. Literature triage (only items relevant to our idea)

| # | Paper | Why it matters here |
|---|---|---|
| L1 | Sampson & Melchior, **"Dynamics of Learning: Generative Schedules from Latent ODEs"**, arXiv 2509.23052 (NeurIPS 2025) | Closest analog. Latent-ODE on `(loss, val_acc, lr)` time series from a hyperparam sweep; at deploy samples a latent ensemble and **picks the LR of the trajectory whose forecast `val_acc(t+Δt)` is highest**. SOTA on CIFAR-100/ResNet18 (73.9 → 74.5%). Key tricks: **no internal model state as input** (generalizes), **goal-conditioned on long-horizon validation**, ensemble + similarity rejection instead of hard argmin. |
| L2 | Dong et al., **AdaLRS: Loss-Guided Adaptive LR Search**, arXiv 2506.13274 (NeurIPS 2025) | Theorem: under mild assumptions, **loss-descent velocity is convex in LR with the same optimum as loss itself**. Algorithm: probe at α·η and β·η, accept the change only if velocity improves by more than the noise floor 2e, plus a **revert + boundary safeguard** if loss rises in two consecutive windows. Almost a drop-in replacement for the "veto" rule `final_resume.md` already calls for. |
| L3 | Subramanian et al., **GreedyLR**, arXiv 2512.14527 | Trivial scheduler: multiply LR by F<1 if val loss worsens, divide by F if it improves. Convergence proof, optimal F = 1 − 1/L_smooth. Reportedly beats cosine in 86% of <500M-param runs. **Use as a 4th baseline** in 6-seed bundles. |
| L4 | Defazio et al., **Schedule-Free AdamW / SGD**, NeurIPS 2024 (winner of MLCommons AlgoPerf 2024 Self-Tuning) | Removes the schedule via specific iterate-averaging / momentum. Strong empirical baseline; repo currently has **no schedule-free baseline**. |
| L5 | Mishchenko & Defazio, **Prodigy**, ICML 2024 (and **D-Adaptation**, ICML 2023) | Estimates distance-to-solution `D` to set LR adaptively without tuning. Strong **second baseline** alongside Schedule-Free. |
| L6 | Chu, Gao, Ye, Udell, **"Provable and Practical Online LR Adaptation with Hypergradient Descent"**, ICML 2025, arXiv 2502.11229 | Resurrects Baydin et al. 2017 hypergradient with proper convergence theory + heavy-ball/Nesterov variants (HDM-HB). Cheap, online, no offline pretraining. Plug in as `--scheduler hypergrad`. |
| L7 | Wu et al., **VolSched**, arXiv 2507.10575 | LR up when long-/short-term volatility ratio shrinks (escape plateaus), down when it grows. +1.3-1.4 pp on CIFAR-100/ResNet, finds **38% flatter minima** than baseline. The volatility ratio is a ~5-line feature — easy to add to the policy input. |
| L8 | Cui et al., **"Dynamic LR for Deep RL: A Bandit Approach"**, arXiv 2410.12598 | Frames LR selection as a **non-stationary multi-armed bandit over an LR grid**. Direct reformulation of `curve_argmin` head: instead of MSE on the curve, regret-minimization over LR_GRID with EXP3-style updates. Naturally avoids saturation because reward is one-step actual progress, not a 1-step probe. |
| L9 | Yu et al., **SALR: Sharpness-Aware LR Scheduler**, IEEE 2023 | Increases LR locally where the loss surface is sharp (ratio of dominant Hessian eigenvalue to gradient norm). The **flatness/sharpness signal can be extracted from the existing 30-dim probe** — see proposal P5. |
| L10 | Lyu et al., **Decoupled Relative LR Schedules (RLRS)**, arXiv 2507.03526 | Different multipliers per Transformer block. Not directly applicable to ResNet-110, but the **per-layer multiplier head** generalizes — predict per-stage multipliers on PreResNet stages instead of one global LR. |

Older L2O work (Andrychowicz 2016, Wichrowska 2017, Metz 2020) is intentionally
out of scope: 2024-2025 reviews agree that learned optimizers still don't
generalize OOD, and our features are richer than what those papers used.

---

## 3. Concrete proposals, ranked by expected ROI

Ordered by "how likely to fix the *specific* failure modes in
`2h_curve/RESULTS.md`", not by how flashy.

### P1. Replace argmin with a goal-conditioned, validation-anchored selector (from L1)

**Fix for failure #1.** Today `deploy_curve_policy_step` picks `argmin` of the
**predicted 1-step loss curve**. As observed, that argmin walks off to LR=2.5
around epoch ~15 because at low gradient norm the 1-step probe is essentially
flat. Sampson & Melchior dodge this exact failure by **conditioning on
long-horizon validation**, not on instantaneous loss.

Implementation:

1. Extend `CurveLRPolicy.head` to additionally output a **predicted
   Δ-validation-accuracy at t+H** for each LR in `LR_GRID` (second output head
   of shape `[B, lookahead_n, landscape_dim]`).
2. Train it offline on the same oracle dataset; `targets_v3_*.npy` already
   contains per-trajectory val-acc — supervise the val-acc head on
   `acc(t+H) − acc(t)` per `(LR, t)`.
3. At deploy, pick `argmax_{LR} predicted Δval_acc(t+H, LR)` instead of
   `argmin curve`.
4. Use an ensemble over latent perturbations (Section 3.2 of L1) and the
   **similarity-rejection rule** ("only accept rollouts whose predicted current
   loss is within 2σ of the actual current loss") — main trick to avoid
   catastrophic actions.

Single change most likely to flip the verdict from NULL to PASS on the curve
policy.

### P2. Add a "veto" / safeguard rule (from L2 and `final_resume.md`)

Already listed as a recommended next step ("validation-time rule that prevents
late-epoch LR increases unless the landscape probe shows a clear improvement").
AdaLRS (L2) gives a **principled way** to do it:

- After applying `next_lr`, measure 1-window loss-descent velocity `v(η_t)`.
- If `v` does not improve by more than `2e` (estimation error of `v`) over the
  cosine reference value, **revert** to cosine for that step and downscale LR
  by `β = 1/max(λ_t · 1.5, 1)`.
- Boundary safeguard: if loss increases for two consecutive windows,
  **clamp to 0.7 × cosine**.

Drop-in to `deploy_curve_policy_step` — does not change the policy net, just
the controller around it. ~30 lines, will turn the catastrophic 32% into
at-worst the cosine baseline.

### P3. Re-parameterize the policy output as a discrete bucket / residual

Combine three ideas into one head:

- **Discrete head** over `LR_GRID` (30-way classification), trained with
  cross-entropy on the oracle's argmin index. Kills the
  regression-to-saturation problem.
- **Residual on cosine**: output `Δlog(LR) ∈ [−1, +1]`, applied as
  `LR = cosine(t) · exp(clip(Δlog, −1, 1))`. Naturally bounded,
  identity-initialized, gradient-friendly.
- Train both heads jointly with a **load-balancing or entropy-regularized**
  auxiliary loss on the discrete head (the bandit framing in L8 has the right
  form).

Code locations: `transform_policy_targets` in `trainer/my_research.py` already
does the cosine_multiplier transform — extend it to emit either the bucket
index or the residual; add a corresponding `policy_output ∈ {bucket,
residual_log}` mode.

### P4. Replace the offline + online curve regression with a non-stationary bandit (from L8)

The current online MPC step in `deploy_curve_policy_step` is essentially **MSE
on a forecast that is never acted on at high resolution**. A bandit
reformulation:

- Treat each LR in `LR_GRID` as an arm.
- Reward `r_t(arm) = -[Δval_acc over next K epochs after applying arm]`.
  Computed counterfactually for the chosen arm; un-chosen arms still get the
  policy's predicted reward as a regression target — gives a hybrid
  online-bandit + supervised loss like Decision Transformer training.
- EXP3-style or LinUCB-style update on top of the GRU's last-hidden as context.

Decouples "predict the curve well" from "pick the right LR", which is exactly
where the current pipeline breaks: curve MSE drops 88% while argmin keeps
drifting.

### P5. Add cheap, high-signal features that the current probe is missing (from L7, L9)

The 30-dim probe is rich, but has three known weaknesses fixable in
`prepare_policy_feature` for almost free:

| Feature | Why | Cost |
|---|---|---|
| **Volatility ratio** = std(loss, window=3) / std(loss, window=15) (L7) | Direct signal of plateau vs noise. VolSched gets +1.4 pp on CIFAR-100 from this alone. | 2 lines |
| **Sharpness proxy** from the 30-dim probe itself: 2nd derivative of curve at current LR (L9) | Tells the policy whether it's in a sharp valley (don't increase LR) or a flat plateau (escape). Curve already exists, this is one finite difference. | 3 lines |
| **Gradient-norm decay rate** over window | Mechanism behind late-training argmin drift is `‖∇L‖→0` flattening the probe. Feeding it explicitly lets the policy *know* the probe is unreliable. | 1 line in `evaluate(...)` |

Three features, zero extra GPU, give the policy information it currently has
to infer indirectly.

### P6. Sanity-baseline the repo against three strong adaptive optimizers (L4, L5, L6)

Right now the only baseline in `results/final_resume.md` is hand-tuned cosine.
Add three near-zero-effort baselines:

1. **Schedule-Free SGD/AdamW** (`pip install schedulefree`, ~10 lines in
   `main()` of `my_research.py`) — winner of MLCommons AlgoPerf 2024.
2. **Prodigy** (`pip install prodigyopt`) — parameter-free, beats hand-tuned
   Adam on ViT/RoBERTa.
3. **Hypergradient SGD (HDM-HB)** (L6) — ~50 lines, online, no offline
   pretraining, paper at ICML 2025 has reference impl.

If the `cosine_multiplier` policy already matches cosine, the right comparison
is "does it beat Schedule-Free / Prodigy / Hypergradient too?". Without those
numbers, even a good result is hard to publish.

### P7. (Speculative, longer term) Decoupled per-block LR multipliers (from L10)

`DynamicScheduler` updates **all param groups with the same LR**. RLRS (L10)
shows on Transformers that **different blocks want very different LR
multipliers** (+23% throughput from this alone). On ResNet-110 there are 3
natural stages × 36 layers each. Cheapest version: have the policy predict
**3 multipliers** instead of 1 (one per stage), apply them to corresponding
param groups in `student_optimizer`.

Requires changing `DynamicScheduler.set_lr` to accept a list and registering
per-stage param groups (~80 lines). Upside is large because the policy gets
strictly more degrees of freedom over the same probe.

---

## 4. Suggested execution order for the A100 budget

6-seed parallel A100 bundle, 35-epoch CIFAR-100 runs:

| Phase | What | GPU cost | Decision gate |
|---|---|---|---|
| **0** | Add Schedule-Free + Prodigy + Hypergradient baselines (P6) | 6 seeds × 3 baselines × 35 epochs ≈ 1 bundle | Establishes the real ceiling. If one of these is at 71%+ already, all effort should go into beating it, not into beating cosine. |
| **1** | P2 (AdaLRS-style veto rule on top of current `curve_argmin`) | 1 bundle | If `curve_argmin` jumps from 32% → ≥69%, the rest of the pipeline is salvageable. If not, abandon `curve_argmin` and go to P3/P4. |
| **2** | P5 (3 cheap features) + P3 (residual-on-log-cosine head) on the existing GRU policy | 2 bundles (with/without P5 ablation) | Goal: beat cosine by ≥0.5 pp at p<0.1. |
| **3** | P1 (val-acc forecast head + ensemble selector) — needs re-running `generate_oracle_dataset.py` to log val-acc curves per LR | offline policy retrain (cheap) + 1 bundle | Swing-for-the-fences experiment that, if it works, gives a SOTA-vs-LODE comparison. |
| **4** (optional) | P4 (bandit reformulation) and/or P7 (per-stage LR) | 2 bundles | Only if Phase 3 has shown the predictions carry signal. |

---

## 5. Two-line summary

**The repo's idea is well-aligned with two 2025 papers: AdaLRS (L2, gives the
principled veto rule we already want) and Latent-ODE LODE (L1, gives the right
way to use the forecast — argmax of predicted future val-acc, not argmin of
the 1-step loss probe).** Implementing P1 + P2 + P5 + adding the
Schedule-Free/Prodigy baselines (P6) is the smallest set of changes that has a
credible chance of turning current NULL verdicts into a publishable PASS.

Cheapest first wins: **P2 (AdaLRS veto in `deploy_curve_policy_step`)** and
**P6 (Schedule-Free baseline in `main()`)** — each is a self-contained patch
on top of current files.
