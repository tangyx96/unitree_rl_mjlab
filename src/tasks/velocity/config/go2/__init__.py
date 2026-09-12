"""Go2速度跟踪任务注册。

将 Unitree-Go2-Rough 和 Unitree-Go2-Flat 两个任务注册到全局任务注册表。
每个任务注册：task_id、env_cfg（训练用）、play_env_cfg（推理用）、rl_cfg（RL超参）、runner_cls（训练器类）。
"""
from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_go2_flat_env_cfg,
  unitree_go2_rough_env_cfg,
)
from .rl_cfg import unitree_go2_ppo_runner_cfg

# 粗糙地形速度跟踪任务
register_mjlab_task(
  task_id="Unitree-Go2-Rough",
  env_cfg=unitree_go2_rough_env_cfg(),
  play_env_cfg=unitree_go2_rough_env_cfg(play=True),  # play模式：无限episode、关噪声、关课程
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# 平坦地形速度跟踪任务
register_mjlab_task(
  task_id="Unitree-Go2-Flat",
  env_cfg=unitree_go2_flat_env_cfg(),
  play_env_cfg=unitree_go2_flat_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)