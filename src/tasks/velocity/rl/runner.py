"""速度跟踪任务的 On-policy 训练器。

在 MjlabOnPolicyRunner 基础上，每次保存 checkpoint 时自动：
1. 导出 ONNX 策略模型（供 C++ 实机部署使用）
2. 将环境元数据（关节映射、PD 增益等）嵌入 ONNX 文件
3. 若使用 wandb 日志，同步上传 ONNX 至云端
"""
import os

import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  """速度跟踪训练器，保存时自动导出部署用 ONNX 模型。"""

  env: RslRlVecEnvWrapper  # 类型标注，不赋值；self.env 由父类 __init__(env, ...) 挂上同一包装器引用

  def save(self, path: str, infos=None):
    """保存 checkpoint 并导出 ONNX 模型。

    Args:
      path: checkpoint 保存路径，通常包含 "model" 关键字，
            以此为界截取前缀作为 ONNX 导出目录。
      infos: 传递给父类的附加信息。
    """
    super().save(path, infos)

    # 从 checkpoint 路径提取实验目录前缀（截掉 "model_*" 部分）
    policy_path = path.split("model")[0]
    filename = "policy.onnx"

    # 将 Actor 网络导出为 ONNX 格式
    self.export_policy_to_onnx(policy_path, filename)

    # 获取 wandb 运行名称，非 wandb 模式下标记为 "local"
    run_name: str = (
      wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]

    # 将环境配置元数据嵌入 ONNX，部署端据此初始化控制器
    onnx_path = os.path.join(policy_path, filename)
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    attach_metadata_to_onnx(onnx_path, metadata)

    # wandb 模式下自动同步 ONNX 文件至云端
    if self.logger.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))