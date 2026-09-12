"""Script to train RL agent with RSL-RL.

训练入口脚本。

用法: python scripts/train.py Unitree-Go2-Flat --env.scene.num-envs=4096

工作流: Train → Play → Sim2Real
本脚本负责 Train 阶段：从任务注册表加载训练用 env_cfg / rl_cfg，创建仿真环境，启动 PPO。
"""

import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import tyro  # CLI参数解析（基于dataclass声明式定义）

# ── mjlab：基于 MuJoCo 的 RL 环境框架（外部依赖，pip install mjlab==1.2.0） ──
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
# ManagerBasedRlEnv:  核心环境类，内部管理观测/动作/奖励/终止/事件/课程/命令等 Manager
# ManagerBasedRlEnvCfg: 环境配置数据类，定义完整MDP（观测/动作/奖励/终止/事件/课程等）
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
# MjlabOnPolicyRunner:  On-policy训练器，封装rsl_rl的PPO训练循环（采集→更新→保存）
# RslRlBaseRunnerCfg:   训练器配置基类（网络结构/PPO超参/迭代次数/保存间隔等）
# RslRlVecEnvWrapper:   将mjlab环境包装为rsl_rl兼容的VecEnv接口（step/reset/clip_actions）
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
# list_tasks:       列出注册表中所有可用任务ID
# load_env_cfg:     按task_id加载环境配置（本脚本用不带play的训练配置）
# load_rl_cfg:      按task_id加载RL超参配置
# load_runner_cls:  按task_id加载训练器类（如VelocityOnPolicyRunner）
from mjlab.tasks.tracking.mdp import MotionCommandCfg
# MotionCommandCfg: 动作模仿任务的命令配置（定义参考运动轨迹的采样方式，tracking任务专用）
from mjlab.utils.gpu import select_gpus
# select_gpus: gpu_ids 是当前 CUDA_VISIBLE_DEVICES 中的下标；None 为 CPU
from mjlab.utils.os import dump_yaml, get_checkpoint_path
# dump_yaml:          将配置序列化为YAML（保存实验配置快照）
# get_checkpoint_path: 按 load_run / load_checkpoint 在本地日志目录查找checkpoint
from mjlab.utils.torch import configure_torch_backends
# configure_torch_backends: 配置PyTorch计算后端（CUDA/matmul/SDPA优化）
from mjlab.utils.wrappers import VideoRecorder
# VideoRecorder: 环境Wrapper，每N步录制一段RGB视频


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg  # 训练用环境配置（非 play）
  agent: RslRlBaseRunnerCfg  # Runner / PPO 超参
  motion_file: str | None = None  # tracking 参考运动 .npz；velocity 任务忽略
  video: bool = False  # 是否录像（仅 rank 0 写入）
  video_length: int = 200  # 每段录像长度（环境步）
  video_interval: int = 2000  # 每 N 步尝试开始一段录像
  enable_nan_guard: bool = False  # 逐步检测 NaN 并导出仿真状态
  torchrunx_log_dir: str | None = None  # 多卡 worker 日志；未设则 {log_dir}/torchrunx；空串可关闭
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])
  """可见 GPU 的下标列表；"all" 用全部；None 为 CPU。默认 [0]。"""

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    """从注册表加载训练用 env_cfg / rl_cfg。"""
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    return TrainConfig(env=env_cfg, agent=agent_cfg)


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  """核心训练函数：创建环境 → 包装环境 → 创建Runner → 开始训练。

  此函数在单GPU时由launch_training直接调用，
  多GPU时由torchrunx在每个worker进程中调用。
  CUDA_VISIBLE_DEVICES 由 launch_training 事先写好。
  """
  # ── 1. 设备 / seed / rank ──
  # 空串（含未设置时的默认值）视为 CPU；非空则 cuda:{LOCAL_RANK}
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))  # 本机 GPU 序号
    rank = int(os.environ.get("RANK", "0"))  # 全局 rank
    # Set EGL device to match the CUDA device.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    # Set seed to have diversity in different processes.
    seed = cfg.agent.seed + local_rank

  configure_torch_backends()  # CUDA/matmul/SDPA 等后端，训练前统一数值与性能行为

  cfg.agent.seed = seed
  cfg.env.seed = seed

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

  # ── 2. tracking：必须提供本地 --motion-file ──
  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in cfg.env.commands and isinstance(
    cfg.env.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task:
    if not cfg.motion_file:
      raise ValueError("For tracking tasks, --motion-file must be set ...")
    motion_path = Path(cfg.motion_file).expanduser().resolve()
    if not motion_path.exists():
      raise FileNotFoundError(f"Motion file not found: {motion_path}")
    motion_cmd = cfg.env.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = str(motion_path)
    print(f"[INFO] Using motion file: {motion_cmd.motion_file}")

    # Check if motion_file is already set (e.g., via CLI --env.commands.motion.motion-file).
    if motion_cmd.motion_file and Path(motion_cmd.motion_file).exists():
      print(f"[INFO] Using local motion file: {motion_cmd.motion_file}")

  # ── 3. NaN 检测 ──
  # Enable NaN guard if requested.
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  # ── 4. 创建环境 ──
  # rgb_array：render() 返回图像数组供录像；不开 --video 则不渲染
  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )

  # ── 5. 本地 resume：从已有 checkpoint 接着训 ──
  log_root_path = log_dir.parent  # Go up from specific run dir to experiment dir.

  resume_path: Path | None = None
  if cfg.agent.resume:
    # Load checkpoint from local filesystem.
    resume_path = get_checkpoint_path(
      log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
    )

  # ── 6. 录像（仅 rank 0，避免多进程抢写） ──
  # Only record videos on rank 0 to avoid multiple workers writing to the same files.
  if cfg.video and rank == 0:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,  # 每 video_interval 步开始一段
      video_length=cfg.video_length,  # 每段录多少环境步
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  # ── 7. 适配 rsl_rl VecEnv ──
  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

  # dataclass → dict：Runner 吃 dict 超参；dump_yaml 写实验快照
  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  # ── 8. Runner ──
  # runner_cls 是注册表里的训练器类（不是实例）。本仓库 velocity 任务注册的是
  # src.tasks.velocity.rl.VelocityOnPolicyRunner；未注册则退回 MjlabOnPolicyRunner。
  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    runner_cls = MjlabOnPolicyRunner

  runner_kwargs = {}
  # 等价于 VelocityOnPolicyRunner(env, ...)。第一个参数是上面的 RslRlVecEnvWrapper，
  # 由父类 __init__ 赋给 self.env（同一引用，不是拷贝）。learn() 做 PPO 采集/更新，
  # save() 存 checkpoint；本仓库子类每次 save 再导出 policy.onnx。
  runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)

  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))  # 载入权重后再 learn()

  # ── 9. 配置快照（仅 rank 0） ──
  # Only write config files from rank 0 to avoid race conditions.
  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  # ── 10. PPO ──
  # init_at_random_ep_len：各 env 初始进度随机，降低同步 bias
  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )

  env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):
  """启动训练：创建日志目录 → 选择GPU → 单GPU直接运行/多GPU用torchrunx启动。

  GPU运行模式：
  - CPU：select_gpus 返回 None，CUDA_VISIBLE_DEVICES=""
  - 单GPU：直接调用 run_train()
  - 多GPU：torchrunx 每卡一个 worker
  """
  args = args or TrainConfig.from_task(task_id)

  # ── 1. 日志目录 ──
  # Create log directory once before launching workers.
  # logs/rsl_rl/<experiment_name>/<timestamp>[_<run_name>]/
  log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
  log_root_path.resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  # ── 2. GPU ──
  # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  # ── 3. 环境变量（run_train 据此选设备） ──
  # Set environment variables for all modes.
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"  # 无头 GPU 渲染

  # ── 4. 启动 ──
  if num_gpus <= 1:
    # CPU or single GPU: run directly without torchrunx.
    run_train(task_id, args, log_dir)
  else:
    # Multi-GPU: use torchrunx.
    import torchrunx

    # torchrunx redirects stdout to logging.
    logging.basicConfig(level=logging.INFO)

    # Configure torchrunx logging directory.
    # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if args.torchrunx_log_dir is not None:
        # User specified a value via flag (could be "" to disable).
        os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
      else:
        # Default: put logs in training directory.
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    # 本机一卡一进程，各 worker 执行同一 run_train；RANK/LOCAL_RANK 由 torchrunx 注入
    torchrunx.Launcher(
      hostnames=["localhost"],  # 单机，非多机集群
      workers_per_host=num_gpus,  # 与可见 GPU 数一致
      backend=None,  # Let rsl_rl handle process group initialization.
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),  # 把 CUDA/MUJOCO_GL 等传入子进程
    ).run(run_train, task_id, args, log_dir)


def main():
  """CLI入口：两阶段解析 → 启动训练。

  import 把所有任务登记进注册表（菜单，不选定）。本次任务由第一个位置参数决定：
    python scripts/train.py Unitree-Go2-Flat --env.scene.num-envs=4096
  1. 解析 task_id → chosen_task；return_unknown_args 留下的是未吃掉的 argv
     （如 --env.scene.num-envs=4096），不是其余任务ID
  2. 用 remaining_args 解析 TrainConfig；from_task(chosen_task) 只加载这一条配置
  """
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401  # mjlab内置任务
  import src.tasks  # 本仓库任务；副作用：register_mjlab_task

  all_tasks = list_tasks()  # 全部合法 task_id，供第一阶段做选项
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),  # 任务名枚举，只吃第一个位置参数
    add_help=False,
    return_unknown_args=True,  # 未吃掉的 CLI 开关，不是其余 task_id
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,  # 用剩下的开关覆盖配置，chosen_task 不是 argv
    default=TrainConfig.from_task(chosen_task),  # 只加载这一条 env_cfg / rl_cfg
    prog=sys.argv[0] + f" {chosen_task}",  # 帮助里显示 train.py <task_id>
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args  # 已消费

  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
