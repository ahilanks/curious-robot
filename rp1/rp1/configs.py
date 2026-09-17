"""Per-domain RP1 configurations (Table 6 of the paper, LeWM backbone).
Time unit for critic / hindsight offsets = action block (5 primitive steps)."""

COMMON = dict(K=8, horizon=5, action_block=5, cross_prob=0.3, critic_ratio=1, ema=0.005,
              n_step=50, td_batch=1024, hidden=512)

TWOROOM = dict(
    env='swm/TwoRoom-v1', dataset='tworoom', ckpt='lewm-tworooms', action_dim=2,
    callables=[
        {'method': '_set_state', 'args': {'state': {'value': 'pos_agent'}}},
        {'method': '_set_goal_state', 'args': {'goal_state': {'value': 'goal_pos_agent'}}},
    ],
    env_kwargs={},
    horizons=[25, 100],
    offline_value=dict(gamma=1.0, expectile=0.1, n_step=50, steps=6000, batch=1024, lr=1e-3),
    critic=dict(gamma=1.0, expectile=0.1, expectile_final=0.1, lr=1e-3, lr_final=1e-3, live_steps=6400,
                value_expansion=0.0),
    actor={  # per goal horizon
        25: dict(amax=1.8, lambda_mean=0.1, actor_lr=1e-4, lr_schedule='const', batch_size=128, steps=8000,
                 replay_prob=0.0, max_delta=12),
        100: dict(amax=2.6, lambda_mean=0.3, actor_lr=1e-3, lr_schedule='const', batch_size=128, steps=8000,
                  replay_prob=0.0, max_delta=12),
    },
    standardize_latents=False,
)

REACHER = dict(
    env='swm/ReacherDMControl-v0', dataset='reacher', ckpt='lewm-reacher', action_dim=2,
    callables=[
        {'method': 'set_state', 'args': {'qpos': {'value': 'qpos'}, 'qvel': {'value': 'qvel'}}},
        {'method': 'set_target_qpos', 'args': {'target_qpos': {'value': 'goal_qpos'}}},
    ],
    env_kwargs=dict(task='qpos_match'),
    horizons=[25],
    offline_value=dict(gamma=0.98, expectile=0.05, n_step=50, steps=6000, batch=1024, lr=1e-3),
    critic=dict(gamma=0.98, expectile=0.1, expectile_final=0.03, lr=1e-3, lr_final=1e-4, live_steps=500,
                value_expansion=0.0),
    actor={
        25: dict(amax=2.2, lambda_mean=0.3, actor_lr=1e-4, lr_schedule='cosine', batch_size=128, steps=1000,
                 replay_prob=0.5, max_delta=12),
    },
    standardize_latents=True,
)

CUBE = dict(
    env='swm/OGBCube-v0', dataset='cube_single_expert', ckpt='lewm-cube', action_dim=5,
    callables=[
        {'method': 'set_state', 'args': {'qpos': {'value': 'qpos'}, 'qvel': {'value': 'qvel'}}},
        {'method': 'set_target_pos', 'args': {'cube_id': {'value': 0, 'in_dataset': False},
                                              'target_pos': {'value': 'goal_privileged_block_0_pos'},
                                              'target_quat': {'value': 'goal_privileged_block_0_quat'}}},
    ],
    env_kwargs=dict(env_type='single', ob_type='states', multiview=False, width=224, height=224,
                    visualize_info=False, terminate_at_goal=True),
    horizons=[25, 100],
    offline_value=dict(gamma=0.98, expectile=0.03, n_step=50, steps=12000, batch=1024, lr=1e-3),
    critic=dict(gamma=0.98, expectile=0.1, expectile_final=0.03, lr=1e-3, lr_final=1e-4, live_steps=3000,
                value_expansion=1.0),
    actor={
        25: dict(amax=1.6, lambda_mean=0.1, actor_lr=3e-4, lr_schedule='cosine', batch_size=256, steps=6000,
                 replay_prob=0.5, max_delta=10),
        100: dict(amax=1.6, lambda_mean=0.1, actor_lr=3e-4, lr_schedule='cosine', batch_size=256, steps=6000,
                  replay_prob=0.5, max_delta=10),
    },
    standardize_latents=False,
)

DOMAINS = dict(tworoom=TWOROOM, reacher=REACHER, cube=CUBE)
