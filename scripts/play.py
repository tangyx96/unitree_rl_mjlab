"""Script to play RL agent with RSL-RL.

推理/回放入口脚本。

用法: python scripts/play.py Unitree-Go2-Flat [--agent trained --checkpoint-file PATH]

加载策略或 dummy 动作，在 MuJoCo 中可视化。agent 三选一：
trained（checkpoint 策略）、zero（全零动作）、random（随机动作）。
"""

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro  # CLI参数解析（基于dataclass声明式定义）

# ── mjlab：基于 MuJoCo 的 RL 环境框架（外部依赖，pip install mjlab==1.2.0） ──
from mjlab.envs import ManagerBasedRlEnv
# ManagerBasedRlEnv: 核心环境类，内部管理观测/动作/奖励/终止/事件/课程/命令等 Manager
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
# MjlabOnPolicyRunner: On-policy训练器，封装rsl_rl的PPO训练循环，play时用于加载checkpoint推理
# RslRlVecEnvWrapper:  将mjlab环境包装为rsl_rl兼容的VecEnv接口（step/reset/clip_actions）
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
# list_tasks:       列出注册表中所有可用任务ID
# load_env_cfg:     按task_id加载环境配置（play=True时取注册的play_env_cfg）
# load_rl_cfg:      按task_id加载RL超参配置
# load_runner_cls:  按task_id加载训练器类（如VelocityOnPolicyRunner）
from mjlab.tasks.tracking.mdp import MotionCommandCfg
# MotionCommandCfg: 动作模仿任务的命令配置（定义参考运动轨迹的采样方式，tracking任务专用）
from mjlab.utils.os import get_wandb_checkpoint_path
# get_wandb_checkpoint_path: 从wandb云端下载并返回checkpoint路径
from mjlab.utils.torch import configure_torch_backends
# configure_torch_backends: 配置PyTorch计算后端（CUDA/matmul/SDPA优化）
from mjlab.utils.wrappers import VideoRecorder
# VideoRecorder: 环境Wrapper，按step_trigger录制RGB视频
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
# NativeMujocoViewer: MuJoCo原生OpenGL渲染窗口（本地可视化，需显示器）
# ViserPlayViewer:    基于Viser的Web远程可视化（浏览器访问，无需本地显示器）


@dataclass(frozen=True)
class PlayConfig:
  """推理/可视化配置。

  三种agent模式：
  - "trained": 加载checkpoint中的Actor网络，确定性推理
  - "zero":    输出全零动作（关节位置动作下通常对应默认关节目标）
  - "random":  输出[-1, 1)均匀随机动作
  """
  agent: Literal["zero", "random", "trained"] = "trained"
  checkpoint_file: str | None = None  # 本地 .pt；trained 且未设时代码会读 wandb_run_path（本 dataclass 未声明该字段）
  motion_file: str | None = None  # tracking 参考运动 .npz；velocity 任务忽略
  num_envs: int | None = None  # None 则沿用 play_env_cfg.scene.num_envs
  device: str | None = None  # None 则 cuda:0（若可用）否则 cpu
  video: bool = False  # 仅 trained 会真正录像（dummy 无 log_dir）
  video_length: int = 200  # 录像片段长度（环境步）
  video_height: int | None = None  # 覆盖 env_cfg.viewer.height
  video_width: int | None = None  # 覆盖 env_cfg.viewer.width
  camera: int | str | None = None  # 本脚本未读取；相机由 env_cfg.viewer 决定
  viewer: Literal["auto", "native", "viser"] = "auto"  # auto：有 DISPLAY/WAYLAND 用 native，否则 viser
  no_terminations: bool = False  # dummy/trained 均生效，便于长时间观看
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False  # CLI 隐藏；tracking 下改为均匀采样运动


def run_play(task_id: str, cfg: PlayConfig):
  """核心play函数：加载环境配置(play模式) → 创建环境 → 加载策略 → 启动可视化。

  流程：
  1. 加载 play_env_cfg / rl_cfg
  2. tracking：绑定参考运动
  3. trained：定位 checkpoint
  4. 创建环境（可选录像）
  5. 构造策略
  6. 启动 viewer
  """
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  # ── 1. 加载配置 ──
  # play=True 取出注册时的 play_env_cfg（任务自行定制，常见：极长 episode、关观测噪声与课程）
  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  DUMMY_MODE = cfg.agent in {"zero", "random"}  # dummy模式：不加载策略，用零/随机动作
  TRAINED_MODE = not DUMMY_MODE  # trained模式：加载checkpoint中的策略

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # ── 2. tracking 任务 ──
  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      # dummy 无本地文件时要求 cfg.registry_name（本 dataclass 未声明该字段）
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )

  # ── 3. 定位 checkpoint（仅 trained） ──
  # 本地 --checkpoint-file；未设时走 cfg.wandb_run_path（字段未在 PlayConfig 声明）
  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent  # 录像等输出写在 checkpoint 同目录

  # ── 4. 创建环境 ──
  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  # dummy 无 log_dir，无法落盘，故不启用 rgb_array
  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,  # 仅全局第 0 步开启一次，录 video_length 步
      video_length=cfg.video_length,
      disable_logger=True,
    )

  # 用RslRlVecEnvWrapper包装，使环境兼容rsl_rl的VecEnv接口
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  # ── 5. 策略 ──
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        """全零动作；关节位置动作下通常对应默认关节目标。"""
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        """[-1, 1) 均匀随机动作。"""
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    # trained：从checkpoint加载Actor，取确定性推理策略
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner  # 优先使用任务注册的Runner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)  # 确定性策略，无探索噪声

  # ── 6. 可视化 ──
  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy).run()  # 本地 OpenGL 窗口，阻塞
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy).run()  # Web 可视化，浏览器访问（通常 localhost:8080）
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401  # mjlab内置任务
  import src.tasks  # 本仓库任务（velocity / tracking）

  # tyro两阶段解析：先解析任务ID，剩余参数留给PlayConfig
  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),  # 将任务ID列表转为Literal类型供tyro解析
    add_help=False,
    return_unknown_args=True,  # 返回未识别的参数（即PlayConfig的参数）
    config=mjlab.TYRO_FLAGS,
  )

  # 第二段只解析 PlayConfig，不覆盖 env_cfg / agent_cfg。
  # 此处 load_rl_cfg 的结果随即丢弃；真正加载在 run_play 内。
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,  # 用第一步剩余的参数解析PlayConfig
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",  # 帮助信息显示为 "play.py Unitree-Go2-Flat"
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
