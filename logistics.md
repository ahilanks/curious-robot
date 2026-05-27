# Results

Findings from training runs. The question each run answers: **does the arm interact
with the blocks and get curious like a baby, and is it learning dynamics?**

What to read off the logs (W&B project `curious-robot`):
- **Interacts with blocks** — `interact/contacts_per_step`, `interact/object_motion` rising over training.
- **Curious / exploring** — `reward/r_cur` staying non-trivial; rollout videos showing varied reaching.
- **Learning dynamics** — `wm/pred_loss` falling *below* `wm/identity_baseline` (persistence); `wm/h_fwd` curriculum advancing. Confirm offline with `src/eval_predictor.py` (pred_mse / persist_mse < 1).
- **Healthy representation** — `encoder/z_std` not collapsing to 0.

## Sweeps

The README `?` constants are swept here before being pinned.

| run | β (sigreg) | λ_cur | δ (deadband) | steps | contacts/step | pred/persist | notes |
|-----|-----------|-------|--------------|-------|---------------|--------------|-------|
| _tbd_ | 0.9 | 15.0 | 0.05 | | | | baseline defaults (λ_cur≈15 → safety:curiosity ~0.5:1) |

## Ablations

| ablation | flag | default | hypothesis / what to compare |
|----------|------|---------|------------------------------|
| PER replay priority | `--per-priority {curiosity,td}` | `curiosity` | `td` = \|TD-error\| is sign-agnostic, so it also replays the unsafe (very-negative `r_safe`) transitions the critic mispredicts — which curiosity priority under-samples once the world model has learned them. **Q:** does `td` better suppress high-safety-penalty / motor-fighting states without losing block interaction or curiosity? Compare `interact/contacts_per_step`, `reward/safe_cur_ratio`, `sac/critic_loss`, and `eval_predictor.py`. Curiosity stays the *reward* either way — this only changes which transitions the critic re-studies. |

## Log

_(add dated entries per run)_
