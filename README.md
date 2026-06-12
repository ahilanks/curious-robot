<!-- ⚠️ Do not modify this README.md without explicit approval from the maintainer.
     AI assistants / contributors must ASK before changing anything in this file. -->

# Curious Robot

JEPA + SIGReg world model with SAC. Everything is trained from scratch. Uses the Mujoco environment with a SO101 6-DOF arm.

## State encoder

$$z_t=\mathrm{MLP}\big(\,\mathrm{MLP}(\mathrm{ViT}(o_t)_{\text{cls}})\ \big\Vert\ \mathrm{MLP}(\mathrm{symlog}(q_t,\dot q_t,u^{\text{app}}_{t-1}))\,\big)\in\mathbb{R}^{256}\quad(192+64)$$

No final per-sample LayerNorm on $z_t$: it would pin every latent to a fixed-radius sphere, fighting SIGReg's isotropic-Gaussian objective (LeWM keeps only BatchNorm inside the projector MLP, no output norm).

## Dynamics + world-model loss

$$\hat z_{t+1}=f\big(z_{t-H_{\text{bwd}}+1:t},\,a_t\big)$$

$$\mathcal{L}_{\text{wm}}=\underbrace{\frac{1}{\sum_{k=1}^{H_{\text{fwd}}}\gamma_{\text{wm}}^{k}}\sum_{k=1}^{H_{\text{fwd}}}\gamma_{\text{wm}}^{k}\,\tfrac{1}{d_z}\lVert \hat z_{t+k}-z_{t+k}\rVert_2^2}_{\text{autoregressive rollout (LeWM plain MSE, no symlog): }\hat z\text{ re-fed with real }a_{t+k}}+\ \beta\,\mathrm{SIGReg}(z_{\text{batch}})$$

Rollout-on variant (vs. the single-horizon loss that scores one sampled $h$ against the real target). $H_{\text{fwd}}$ starts at $1$ and bumps to $H_{\text{fwd}}{+}1$ when the pred term's mean relative change over the last $200$ WM updates falls below tol, capped at $H_{\text{fwd,max}}$ (default $1$ — curriculum off; raise to re-enable, e.g.\ $20$).

## Actor + actuation

$$a_t=\pi(z_t)=\tanh(\mathrm{MLP}(z_t))\in(-1,1)^n,\quad \Delta q_t=a_t\odot\Delta q^{\max}$$

The policy is **deterministic** (stochasticity removed 2026-06-12): no Gaussian head, no sampling, no entropy bonus — exploration is driven by the curiosity reward, not action noise.

$$\tau_t=\mathrm{clip}\!\big(K_p\,\Delta q_t-K_d\,\dot q_t,\ -\tau_i^{\max},\ \tau_i^{\max}\big)$$

## Rewards

$$r_t^{\text{cur}}=\tfrac{1}{d_z}\lVert \hat z_{t+1}-z_{t+1}\rVert_2^2,\qquad
r_t^{\text{safe}}=-\sum_{i=1}^{n}\frac{|\tau_{i,t}|}{\tau_i^{\max}}\,\max\!\big(0,\,-\tau_{i,t}\,\ddot q_{i,t}-\delta\big),\quad \ddot q_{i,t}=\frac{\dot q_{i,t}-\dot q_{i,t-\Delta t_{\text{safe}}}}{\Delta t_{\text{safe}}}$$

$$r_t=\lambda_{\text{safe}}\,r_t^{\text{safe}}+\lambda_{\text{cur}}\,\mathrm{symlog}\!\big(r_t^{\text{cur}}\big),\qquad (o_{t+1},\,r_t^{\text{safe}})=\mathrm{Env}(a_t)$$

The curiosity error is the **per-dim mean** squared error $\tfrac{1}{d_z}\lVert\cdot\rVert_2^2$ (same normalization as $\mathcal L_{\text{wm}}$), so $r^{\text{cur}}\!\sim\!O(0.1\text{–}1)$ and $\mathrm{symlog}$ stays in its discriminative region. The safety penalty carries its own weight $\lambda_{\text{safe}}$ (not pinned to $1$). With the real-arm-calibrated deadband ($\delta=9$, 2026-06-12: all benign motion incl. max-violence reversals scores $\le7.4$, user-labeled-bad events $\ge10.7$ on true-dt $\ddot q$), $r^{\text{safe}}$ is identically $0$ in normal operation — $\lambda_{\text{safe}}$ scales only genuine events, and $2.2$ makes one median bad substep cancel $\approx$ one decision's curiosity term.

## Actor-critic objective (deterministic)

$$y=r_t+\gamma\,(1-d_t)\,\min_i Q_{\bar\theta_i}\big(z_{t+1},\,\pi(z_{t+1})\big)$$

$$\mathcal{L}_Q=\big(Q_\theta(z_t,a_t)-y\big)^2,\qquad
\mathcal{L}_\pi=-\,\min_i Q_i\big(z_t,\,\pi(z_t)\big)$$

TD3-style: twin critics with a Polyak target $Q_{\bar\theta}$ and a deterministic $\pi$ — the SAC entropy terms ($\alpha\log\pi$) and action sampling were removed 2026-06-12. Real $z_{t+1}$, no WM rollout in the target; $d_t$ via truncation-as-done. Buffer holds $(o_t,q_t,\dot q_t,a_t,r_t,o_{t+1})$ for WM + actor-critic; cap $=10\%$ of the run (capped); PER for sampling, FIFO (ring) eviction.

## Constants

| Symbol | Meaning | Value |
|---|---|---|
| $d_z$ ($d_{\text{vis}},d_{\text{prop}}$) | latent state size: ViT branch ⊕ proprio branch | 256 (192, 64) |
| $n$ | joints = action dims per control step | 6 |
| $H_{\text{bwd}}$ | past latents the predictor conditions on | 3 |
| $H_{\text{fwd}}$ / $H_{\text{fwd,max}}$ | $\mathcal L_{\text{wm}}$ rollout length: start / cap | 1 / 1 (curriculum off; was 20) |
| $\gamma_{\text{wm}}$ | discount on far rollout steps in $\mathcal L_{\text{wm}}$ | 0.95 |
| $\beta$ | SIGReg (anti-collapse) weight in $\mathcal L_{\text{wm}}$ | 0.3 |
| batch | minibatch size: WM / SAC updates | 128 / 128 |
| $\gamma$ | actor-critic return discount | 0.9 |
| $\rho$ | Polyak rate of the target critic $Q_{\bar\theta}$ — slows TD-target drift (the critic's target net, **not** a JEPA EMA teacher; SIGReg needs none) | 0.005 |
| $\lambda_{\text{cur}}$ | reward weight on $\mathrm{symlog}(r^{\text{cur}})$ | 15 (since 2026-06-11; safe15 + the earlier campaign ran 20) |
| $\lambda_{\text{safe}}$ | reward weight on $r^{\text{safe}}$ | 2.2 (since 2026-06-12; one median bad substep ≈ one decision's curiosity. Was 0.1) |
| $K_p,K_d$ | sim position-actuator PD gains; hardware reuses them for the obs-torque recompute | 499.11 N·m/rad / 2.731 N·m·s/rad (RBE501 DC-motor model at firmware P=8, since 2026-06-12 — $K_p$ linear in P, was 998.22 at P=16; $K_d$ is back-EMF only, P/D-independent) |
| $\Delta q^{\max}$ | step-size scale: rad of joint delta per unit tanh action (`action_max` in code) | 0.3 sim default; hardware campaign pinned 0.1 |
| $\tau_i^{\max}$ | per-joint torque clip; also normalizes $\|\tau\|/\tau^{\max}$ in $r^{\text{safe}}$ | 3.35 N·m (all joints) |
| $\delta$ | $r^{\text{safe}}$ deadband: $-\tau\,\ddot q$ below it costs nothing | 9 (since 2026-06-12, calibrated on the real arm at P8/D16: benign ≤7.4, labeled-bad ≥10.7. Was 15, which never fired) |
| $\Delta t_{\text{safe}}$ | control-substep period = $\ddot q$ finite-diff window | 0.030 s (frame_skip 6 × 0.005 s sim step). Hardware paces commands to 0.030 s but divides $\ddot q$ by the **measured** read-to-read dt (~0.044 s) since 2026-06-12 |
| action_block | substeps the actor commits per decision (action dim $6{\times}5=30$) | 5 |
| PER $\alpha$ / $\beta_0$ | priority sharpening / importance-weight anneal start | 0.6 / 0.4 |
| flatline window / tol | bump $H_{\text{fwd}}$ when pred-loss relative change < tol over window | 200 / 0.03 |
| lr$_{\text{wm}}$ / lr$_{\text{ac}}$ | learning rates: WM (AdamW) / actor+critic (Adam) | 5e-5 / 3e-4 |
| updates_per_step / wm_update_every | SAC updates per decision / WM update every Nth decision | 1 / 4 |