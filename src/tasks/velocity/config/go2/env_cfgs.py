"""Go2速度跟踪环境配置。

两层定制：
1. unitree_go2_rough_env_cfg() — 在通用工厂基础上定制Go2专属参数（接触传感器、步态、奖励参数等）
2. unitree_go2_flat_env_cfg() — 在rough基础上做减法：去地形、去高度扫描、降碰撞精度
"""

from typing import Literal

from src.assets.robots import (
  get_go2_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

TerrainType = Literal["rough", "obstacles"]


def unitree_go2_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Go2粗糙地形速度跟踪配置。

  在通用工厂make_velocity_env_cfg()基础上定制Go2专属参数：
  - 机器人实体（Go2 MJCF模型+执行器）
  - Raycast传感器frame设为base_link
  - 足部/非足部接触传感器
  - Go2专属奖励参数（步态offset、姿态标准差等）
  - 非法接触终止条件
  """
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500  # Go2需要更多CCD迭代以处理足端碰撞
  cfg.sim.contact_sensor_maxmatch = 500

  cfg.scene.entities = {"robot": get_go2_robot_cfg()}

  # Raycast传感器frame设为Go2的base_link（机身）
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "base_link"

  # Go2四足命名：FR=右前, FL=左前, RR=右后, RL=左后
  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

  # 足部触地传感器：检测四足与地面的接触，用于步态奖励和足底滑行惩罚
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,  # 跟踪腾空/触地时间（critic 的 foot_air_time；步态奖励用触地时间）
  )
  # 非足部触地传感器：检测膝盖/小腿等非足部位触地，用于非法接触终止
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      # Grab all collision geoms...
      pattern=r".*_collision\d*$",
      # Except for the foot geoms.
      exclude=tuple(geom_names),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    nonfoot_ground_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)

  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  cfg.rewards["pose"].params["std_standing"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.05,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.1,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.15,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }

  cfg.rewards["foot_gait"].params["offset"] = [0.0, 0.5, 0.5, 0.0]  # 腿顺序 FR,FL,RR,RL：FR+RL 同相(0)，FL+RR 同相(0.5) → 对角小跑
  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names

  cfg.terminations["illegal_contact"] = TerminationTermCfg(  # 非足部位触地力>10N则终止
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 10.0},
  )

  # Play模式定制：推理/可视化时使用
  if play:
    cfg.episode_length_s = int(1e9)  # 无限episode长度

    cfg.observations["actor"].enable_corruption = False  # 关闭观测噪声
    cfg.events.pop("push_robot", None)  # 关闭随机推力
    cfg.curriculum = {}  # 关闭课程学习
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_go2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Go2平坦地形速度跟踪配置。

  在rough配置基础上做减法：
  1. 地形改为平面（移除地形生成器）
  2. 移除raycast传感器和height_scan观测（平地无需扫描地形）
  3. 移除地形课程学习
  4. 降低碰撞检测参数（平地更简单，无需高精度CCD）
  5. Play模式下缩小速度命令范围
  """
  cfg = unitree_go2_rough_env_cfg(play=play)

  # 降低碰撞检测参数（平地场景更简单）
  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50  # 平地只需50次CCD迭代（rough用500）
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # 切换为平坦地形
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None  # 移除地形生成器

  # 移除raycast传感器和height_scan观测（平地无需扫描地形）
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # 移除地形课程学习（平地无需渐进）
  cfg.curriculum.pop("terrain_levels", None)

  # Play模式下缩小速度命令范围（更安全的演示速度）
  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg