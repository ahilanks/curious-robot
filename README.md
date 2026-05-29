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

Rollout-on variant (vs. the single-horizon loss that scores one sampled $h$ against the real target). $H_{\text{fwd}}$ is **pinned at $1$** (single-step prediction; the flatline curriculum is disabled).

## Actor + actuation

$$a_t^{\text{raw}}\sim\pi(\cdot\mid z_t),\quad a_t=\tanh(a_t^{\text{raw}})\in(-1,1)^n,\quad \Delta q_t=a_t\odot\Delta q^{\max}$$

$$\tau_t=\mathrm{clip}\!\big(K_p\,\Delta q_t-K_d\,\dot q_t,\ -\tau_i^{\max},\ \tau_i^{\max}\big)$$

## Rewards

$$r_t^{\text{cur}}=\tfrac{1}{d_z}\lVert \hat z_{t+1}-z_{t+1}\rVert_2^2,\qquad
r_t^{\text{safe}}=-\sum_{i=1}^{n}\frac{|\tau_{i,t}|}{\tau_i^{\max}}\,\max\!\big(0,\,-\tau_{i,t}\,\ddot q_{i,t}-\delta\big),\quad \ddot q_{i,t}=\frac{\dot q_{i,t}-\dot q_{i,t-\Delta t_{\text{safe}}}}{\Delta t_{\text{safe}}}$$

$$r_t=\lambda_{\text{safe}}\,r_t^{\text{safe}}+\lambda_{\text{cur}}\,\mathrm{symlog}\!\big(r_t^{\text{cur}}\big),\qquad (o_{t+1},\,r_t^{\text{safe}})=\mathrm{Env}(a_t)$$

$\lambda_{\text{safe}}=0$ by default (safety ablated → curiosity-only reward); set $\lambda_{\text{safe}}=1$ to restore the safety term. $r_t^{\text{cur}}$ is the **per-dim mean** squared 1-step prediction error (same normalization as $\mathcal{L}_{\text{wm}}$, so $r_t^{\text{cur}}\sim O(0.1\!-\!1)$), kept small so $\mathrm{symlog}$ operates in its discriminative region rather than its saturated tail. Still symlog-compressed before entering SAC.

## SAC objective &nbsp; [le-wm]

$$y=r_t+\gamma\,(1-d_t)\Big(\min_i Q_{\bar\theta_i}(z_{t+1},a')-\alpha\log\pi(a'\mid z_{t+1})\Big),\quad a'\sim\pi(\cdot\mid z_{t+1})$$

$$\mathcal{L}_Q=\big(Q_\theta(z_t,a_t)-y\big)^2,\qquad
\mathcal{L}_\pi=-\,\mathbb{E}_{a\sim\pi}\big[\min_i Q_i(z_t,a)-\alpha\log\pi(a\mid z_t)\big]$$

Real $z_{t+1}$, no WM rollout in the target; $d_t$ via truncation-as-done. Buffer holds $(o_t,q_t,\dot q_t,a_t,r_t,o_{t+1})$ for WM + SAC; cap $=10\%$ of the run (capped); PER for sampling, FIFO (ring) eviction.

## Constants

| Symbol | Meaning | Value |
|---|---|---|
| $d_z$ ($d_{\text{vis}},d_{\text{prop}}$) | state dim (vis, proprio) | 256 (192, 64) |
| $n$ | DOF / action dim | 6 |
| $H_{\text{bwd}}$ | predictor context (history) | 3 |
| $H_{\text{fwd}}$ / $H_{\text{fwd,max}}$ | rollout horizon: start / max | 1 / 1 (pinned) |
| $\gamma_{\text{wm}}$ | WM rollout discount | 0.95 |
| $\beta$ | SIGReg weight | 0.3 |
| batch | WM batch | 128 |
| $\gamma$ | SAC discount | 0.9 |
| $\alpha$ | entropy coef (fixed) | 0.2 |
| $\rho$ | Polyak target rate | 0.005 |
| $\lambda_{\text{cur}}$ | curiosity weight | 1 |
| $\lambda_{\text{safe}}$ | safety-penalty weight | 0 (ablated) |
| $K_p,K_d$ | PD gains | ? |
| $\Delta q^{\max}$ | per-joint delta clip | large ($\approx\infty$) |
| $\tau_i^{\max}$ | motor torque limit | 2.94–3.35 N·m |
| $\delta$ | safety deadband | ? |
| $\Delta t_{\text{safe}}$ | accel finite-diff window | 10 timesteps |
| action_block | env steps / decision | 5 |
| action_max | actor output scale | 0.3 |
| PER $\alpha$ / $\beta_0$ | priority / IS exponent | 0.6 / 0.4 |
| flatline window / tol | curriculum trigger | 200 / 0.03 |
| lr$_{\text{wm}}$ / lr$_{\text{ac}}$ | AdamW / Adam | 5e-5 / 3e-4 |
| updates_per_step / wm_update_every | SAC updates per decision step / WM update period | 1 / 4 |