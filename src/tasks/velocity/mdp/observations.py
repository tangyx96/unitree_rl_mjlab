"""速度跟踪任务的自定义观测函数。

补充mjlab内置观测之外的任务专属观测：
- foot_height: 足端高度（critic特权信息）
- foot_air_time: 足端腾空时间（critic特权信息）
- foot_contact: 足端接触标志（critic特权信息）
- foot_contact_forces: 足端接触力对数压缩（critic特权信息）
- phase: 步态相位[sin,cos]（actor可用，提供周期性时间信息）
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """足端高度观测：返回各足端site的世界坐标z值。"""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """足端腾空时间观测：自上次离地以来的时间。"""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """足端接触标志观测：1=触地, 0=腾空。"""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """足端接触力观测（对数压缩）：sign(f) * log(1 + |f|)，避免大力值主导。"""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def phase(env: ManagerBasedRlEnv, period: float, command_name: str) -> torch.Tensor:
  """步态相位观测：[sin(2πt/T), cos(2πt/T)]，提供周期性时间信息。

  当命令 [vx, vy, ωz] 的 L2 范数 < 0.1 时相位归零，避免低指令下相位空转。
  """
  global_phase = (env.episode_length_buf * env.step_dt) % period / period
  phase = torch.zeros(env.num_envs, 2, device=env.device)
  phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
  phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
  stand_mask = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
  phase = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)
  return phase