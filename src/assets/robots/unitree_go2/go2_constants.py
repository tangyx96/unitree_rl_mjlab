"""Go2机器人物理常量与模型配置。

定义了Go2的：
- MJCF模型路径和资源加载
- 执行器参数（PD增益、力矩限制、转动惯量）
- 初始关节姿态（站立姿态）
- 碰撞配置（全身碰撞 vs 仅足部碰撞）
- 机器人实体配置工厂函数 get_go2_robot_cfg()
"""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

GO2_XML: Path = (
  SRC_PATH / "assets" / "robots" / "unitree_go2" / "xmls" / "go2.xml"
)
assert GO2_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, GO2_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(GO2_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

# 执行器配置：PD位置控制器，Kp=刚度, Kd=阻尼, effort_limit=最大力矩
GO2_ACTUATOR_HIP = BuiltinPositionActuatorCfg(  # 髋关节：Kp=20, Kd=1, τmax=23.5Nm
  target_names_expr=(
    ".*hip_.*",
  ),
  stiffness=20.0,
  damping=1.0,
  effort_limit=23.5,
  armature=0.01,  # 转子反射惯量
)
GO2_ACTUATOR_THIGH = BuiltinPositionActuatorCfg(  # 大腿关节：Kp=20, Kd=1, τmax=23.5Nm
  target_names_expr=(
    ".*thigh_.*",
  ),
  stiffness=20.0,
  damping=1.0,
  effort_limit=23.5,
  armature=0.01,
)
GO2_ACTUATOR_CALF = BuiltinPositionActuatorCfg(  # 小腿关节：Kp=40, Kd=2, τmax=45Nm（更大增益因承重）
  target_names_expr=(
    ".*calf_.*",
  ),
  stiffness=40.0,
  damping=2.0,
  effort_limit=45,
  armature=0.02,
)

##
# Keyframes.
##


# 初始站立姿态：基座高0.32m，大腿前伸0.9rad，小腿后弯-1.8rad，髋关节微张
INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.32),  # 基座初始位置
  joint_pos={
    ".*thigh_joint": 0.9,  # 大腿前伸
    ".*calf_joint": -1.8,  # 小腿后弯
    ".*R_hip_joint": 0.1,  # 右髋微张
    ".*L_hip_joint": -0.1,  # 左髋微张
  },
  joint_vel={".*": 0.0},  # 初始关节速度为零
)

##
# Collision config.
##

_foot_regex = "^[FR][LR]_foot_collision$"

# 仅足部碰撞：禁用所有碰撞几何，只保留足端碰撞（足端之间也不碰撞）
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_foot_regex,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
  solimp=(0.9, 0.95, 0.023),
)

# 全身碰撞：启用所有碰撞几何（排除自碰撞），足端使用自定义condim/friction/solimp
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={_foot_regex: 3, ".*_collision": 1},
  priority={_foot_regex: 1},
  friction={_foot_regex: (0.6,)},
  solimp={_foot_regex: (0.9, 0.95, 0.023)},
  contype=1,
  conaffinity=0,
)

##
# Final config.
##

GO2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    GO2_ACTUATOR_HIP,
    GO2_ACTUATOR_THIGH,
    GO2_ACTUATOR_CALF,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_go2_robot_cfg() -> EntityCfg:
  """创建Go2机器人配置实例。

  每次调用返回新实例，避免多处共享同一配置时的修改冲突。
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=GO2_ARTICULATION,
  )

if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_go2_robot_cfg())

  viewer.launch(robot.spec.compile())