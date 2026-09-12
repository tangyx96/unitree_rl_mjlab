"""Go2速度跟踪任务 — PPO训练侧配置。

RslRlOnPolicyRunnerCfg 是 mjlab.rl 提供的训练配置 dataclass，
与 ManagerBasedRlEnvCfg（环境/MDP侧）成对注册，CLI 对应 --agent.*。
不含仿真器参数；并行环境数由 scene.num_envs（--num-envs）决定。

三类配置：
  网络 (RslRlModelCfg)
    actor/critic 的 hidden_dims、activation、obs_normalization；
    actor 额外含 distribution_cfg（高斯策略参数）。
  PPO (RslRlPpoAlgorithmCfg)
    学习率与调度、裁剪、熵正则、价值损失、epoch/minibatch、γ/λ、KL、梯度裁剪。
  Runner 调度
    experiment_name、num_steps_per_env、max_iterations、save_interval。
"""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_go2_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Go2速度跟踪PPO配置。"""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),  # 三层全连接隐层，依次 512→256→128 神经元
      activation="elu",  # 隐层间 ELU 非线性；输出头不加激活
      obs_normalization=True,  # 网络输入侧 (o−μ)/σ 归一化，滑动统计均值标准差
      distribution_cfg={
        "class_name": "GaussianDistribution",  # 高斯策略 π(a|o)=N(μ(o),σ)
        "init_std": 1.0,  # 初始标准差
        "std_type": "scalar",  # 各动作维度共享同一 σ
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,  # 价值损失权重
      use_clipped_value_loss=True,  # 裁剪价值损失，防更新过大
      clip_param=0.2,  # PPO 比率裁剪 ε
      entropy_coef=0.01,  # 熵正则系数，鼓励探索
      num_learning_epochs=5,  # 每次迭代更新 epoch 数
      num_mini_batches=4,  # mini-batch 切分数
      learning_rate=1.0e-3,
      schedule="adaptive",  # 按 KL 散度自适应调整学习率
      gamma=0.99,  # 折扣因子
      lam=0.95,  # GAE-λ
      desired_kl=0.01,  # 目标 KL 散度（adaptive 调度用）
      max_grad_norm=1.0,  # 梯度范数裁剪上界
    ),
    experiment_name="go2_velocity",  # 日志目录 logs/rsl_rl/{name}/...
    save_interval=100,  # 每隔此数迭代存 checkpoint
    num_steps_per_env=24,  # 每次迭代每环境采样步数
    max_iterations=10001,  # 最大训练迭代次数
  )