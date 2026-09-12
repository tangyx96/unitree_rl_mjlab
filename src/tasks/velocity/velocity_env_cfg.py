"""速度跟踪任务的通用环境配置工厂。

这是整个项目最核心的文件。make_velocity_env_cfg() 定义了所有机器人共享的 MDP：
- 观测（actor/critic两组，不对称Actor-Critic架构）
- 动作（关节位置控制）
- 命令（均匀速度采样）
- 奖励（15项：速度跟踪+姿态+步态+平滑+惩罚）
- 终止（超时+倾倒）
- 事件（重置+域随机化）
- 课程（地形+速度渐进）

机器人专属配置（如Go2/G1/H1_2）调用此工厂函数后，再覆盖机器人特定参数。
"""

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp  # mjlab 环境层 MDP；本文件只用 envs_mdp.height_scan
from mjlab.envs.mdp import dr  # 域随机化，名字是 dr 不是 mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

# 此后 mdp 只指本仓库包。它在 __init__ 里 from mjlab.envs.mdp import *，
# 因此 mdp.joint_pos_rel 能用，但函数写在 mjlab；mdp.phase 等写在本仓库 mdp/*.py。
import src.tasks.velocity.mdp as mdp


def make_velocity_env_cfg() -> ManagerBasedRlEnvCfg:
  """创建速度跟踪任务的通用环境配置。

  返回的 ManagerBasedRlEnvCfg 包含完整的 MDP 定义，
  后续各机器人配置函数在此基础上定制参数。
  """

  ##
  # 传感器：地形高度扫描（raycast），粗糙地形时使用，平坦地形时会被移除
  ##

  terrain_scan = RayCastSensorCfg(
    name="terrain_scan",  # 传感器名称，观测项 height_scan 通过此名称引用
    frame=ObjRef(type="body", name="", entity="robot"),  # 射线发射原点附着的body，name=""为占位，各机器人配置会覆盖（如Go2→base_link, G1→pelvis）
    ray_alignment="yaw",  # 射线网格跟随机器人偏航角旋转，转弯时扫描区域跟着转；若设为"world"则固定在世界坐标系
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),  # 网格射线图案：1.6m(前后)×1.0m(左右)，间距0.1m，约17×11=187条射线向下发射
    max_distance=5.0,  # 每条射线最大探测距离5m，超出未命中返回此值；观测里再乘 scale=1/5.0（不保证落在[0,1]）
    exclude_parent_body=True,  # 排除射线原点所在body自身，避免射线刚发射就命中机器人自己的几何体
    debug_vis=True,  # 在仿真可视化中绘制射线（调试用）
    viz=RayCastSensorCfg.VizCfg(show_normals=True),  # 可视化时同时显示命中点的法向量，帮助观察地形表面朝向
  )

  ##
  # 观测：不对称Actor-Critic（Asymmetric Actor-Critic）
  #
  # actor 仅使用部署时可获取的观测量（IMU、编码器、命令等），并施加传感器噪声
  # （enable_corruption=True），使策略在真实传感器误差下仍具鲁棒性。
  #
  # critic = actor 全部观测量 + 特权信息（privilege），且不加噪声
  # （enable_corruption=False）。特权信息包括基座线速度、足端高度/接触等，
  # 仿真中精确可得但真机难以获取的量。critic 凭此享有更准确的价值估计，
  # 为 actor 提供更精准的 advantage 信号，加速策略收敛。
  #
  # 注意：critic 通过 **actor_terms 展开后追加同名 key 实现覆盖，
  # 如 height_scan 在 critic 中重新定义为无噪声版本，替换 actor 版本。
  # 部署时仅保留 actor 观测，critic 不参与推理，因此无需特权传感器。
  # 参考: Lee et al., "Learning quadrupedal locomotion over challenging terrain" (Sci. Robot. 2020)
  ##

  actor_terms = {
    "base_ang_vel": ObservationTermCfg(  # IMU角速度（3维），部署时从IMU获取
      func=mdp.builtin_sensor,  # 源码在 mjlab.envs.mdp，经本仓库 mdp 再导出 | 使用内置传感器读取函数
      params={"sensor_name": "robot/imu_ang_vel"},  # 读取robot实体的IMU角速度传感器
      noise=Unoise(n_min=-0.2, n_max=0.2),  # 均匀噪声±0.2，模拟IMU角速度测量误差
    ),
    "projected_gravity": ObservationTermCfg(  # 投影重力向量（3维），表示机身姿态
      func=mdp.projected_gravity,  # 源码在 mjlab.envs.mdp，经本仓库 mdp 再导出 | 将重力向量投影到机身坐标系，反映roll/pitch倾斜
      noise=Unoise(n_min=-0.05, n_max=0.05),  # 均匀噪声±0.05，模拟姿态估计误差
    ),
    "command": ObservationTermCfg(  # 速度命令（3维）：[vx, vy, ωz]
      func=mdp.generated_commands,  # 源码在 mjlab.envs.mdp，经本仓库 mdp 再导出 | 读取命令管理器生成的速度指令
      params={"command_name": "twist"},  # 引用名为"twist"的速度命令
    ),
    "phase": ObservationTermCfg(  # 步态相位（2维）：[sin, cos]，周期0.6s
      func=mdp.phase,  # 源码在 src/tasks/velocity/mdp/observations.py | 生成步态相位信号，帮助策略学习周期性步态
      params={"period": 0.6, "command_name": "twist"},  # 周期0.6s；||cmd||<0.1 时相位置零，周期不随速度缩放
    ),
    "joint_pos": ObservationTermCfg(  # 关节位置相对默认姿态（维数=关节数，Go2 为12）
      func=mdp.joint_pos_rel,  # 源码在 mjlab.envs.mdp，经本仓库 mdp 再导出 | 当前关节位置减去默认站立位置，得到偏差
      noise=Unoise(n_min=-0.01, n_max=0.01),  # 均匀噪声±0.01 rad，模拟编码器位置误差
    ),
    "joint_vel": ObservationTermCfg(  # 关节速度（维数=关节数，Go2 为12）
      func=mdp.joint_vel_rel,  # 源码在 mjlab.envs.mdp，经本仓库 mdp 再导出 | 当前关节速度（相对量，默认偏移为0所以与绝对值相同）
      noise=Unoise(n_min=-1.5, n_max=1.5),  # 均匀噪声±1.5 rad/s，模拟编码器速度误差
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),  # 源码在 mjlab.envs.mdp，经本仓库 mdp 再导出 | 上一步动作（维数=动作维，Go2 为12），提供动作历史
    "height_scan": ObservationTermCfg(  # 地形高度扫描（raycast），平坦地形时会被移除
      func=envs_mdp.height_scan,  # 源码在 mjlab.envs.mdp，此处用 envs_mdp 直引，不经过本仓库 mdp | 从raycast读高度图
      params={"sensor_name": "terrain_scan"},  # 引用名为"terrain_scan"的射线传感器
      noise=Unoise(n_min=-0.1, n_max=0.1),  # 均匀噪声±0.1，模拟地形感知误差
      scale=1 / terrain_scan.max_distance,  # 乘 0.2 缩小量级；height_scan 若为相对高度则可负，不保证[0,1]
    ),
  }

  # critic = actor 观测（**actor_terms 展开）+ 特权信息；同名 key 覆盖 actor 版本
  critic_terms = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(  # 基座线速度（3维），特权：真机无精确线速度传感器（IMU积分漂移）
      func=mdp.builtin_sensor,  # 源码在 mjlab.envs.mdp，经本仓库 mdp 再导出 | 使用内置传感器读取函数
      params={"sensor_name": "robot/imu_lin_vel"},  # 读取robot实体的IMU线速度传感器
      noise=Unoise(n_min=-0.5, n_max=0.5),  # 均匀噪声±0.5（critic enable_corruption=False，此噪声不生效）
    ),
    "height_scan": ObservationTermCfg(  # 覆盖 actor 版本：critic 的地形扫描不加噪声，享有精确地形信息
      func=envs_mdp.height_scan,  # 源码在 mjlab.envs.mdp，此处用 envs_mdp 直引，不经过本仓库 mdp | 从raycast读高度图
      params={"sensor_name": "terrain_scan"},  # 引用名为"terrain_scan"的射线传感器
      scale=1 / terrain_scan.max_distance,  # 乘 0.2 缩小量级，与 actor 相同，但不加噪声
    ),
    "foot_height": ObservationTermCfg(  # 足端世界系 z（维数=足数，Go2 为4），特权：需深度或外部定位
      func=mdp.foot_height,  # 源码在 src/tasks/velocity/mdp/observations.py | 足端 site 的世界坐标 z，不是相对地形高度
      params={"asset_cfg": SceneEntityCfg("robot", site_names=())},  # Set per-robot. 各机器人覆盖site_names
    ),
    "foot_air_time": ObservationTermCfg(  # 足端腾空时间（4维），特权：需接触传感器持续计时
      func=mdp.foot_air_time,  # 源码在 src/tasks/velocity/mdp/observations.py | 每只足自上次离地后的时间（critic 观测，步态奖励不读此项）
      params={"sensor_name": "feet_ground_contact"},  # 引用足部接触传感器
    ),
    "foot_contact": ObservationTermCfg(  # 足端接触标志（4维），特权：需足底触地开关
      func=mdp.foot_contact,  # 源码在 src/tasks/velocity/mdp/observations.py | 二值标志：1=触地，0=腾空
      params={"sensor_name": "feet_ground_contact"},  # 引用足部接触传感器
    ),
    "foot_contact_forces": ObservationTermCfg(  # 足端接触力（对数压缩，4*3=12维），特权：需六维力传感器
      func=mdp.foot_contact_forces,  # 源码在 src/tasks/velocity/mdp/observations.py | 读取足端接触力并做对数压缩，避免大力值主导梯度
      params={"sensor_name": "feet_ground_contact"},  # 引用足部接触传感器
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(  # actor观测组：部署时可用，启用观测噪声模拟传感器误差
      terms=actor_terms,
      concatenate_terms=True,  # 将所有观测项拼接为一个一维向量输入网络
      enable_corruption=True,  # 启用噪声腐蚀（模拟真实传感器噪声）
      history_length=1,  # 观测历史长度1（仅当前步，不堆叠历史帧）
    ),
    "critic": ObservationGroupCfg(  # critic观测组：仅训练时可用，不加噪声
      terms=critic_terms,
      concatenate_terms=True,  # 将所有观测项拼接为一个一维向量输入网络
      enable_corruption=False,  # 不启用噪声腐蚀，critic享有精确特权信息
      history_length=1,  # 观测历史长度1（仅当前步，不堆叠历史帧）
    ),
  }

  ##
  # Metrics
  ##

  metrics = {
    "mean_action_acc": MetricsTermCfg(  # 平均动作加速度，衡量动作平滑度
      func=mdp.mean_action_acc,  # 源码在 mjlab.envs.mdp，经本仓库 mdp 再导出 | 平均动作加速度
    ),
  }

  ##
  # 动作：关节位置控制，目标 = 默认关节角 + scale * 网络输出
  # scale=0.25：输出再乘 0.25 弧度；高斯策略本身不保证落在 [-1, 1]
  # use_default_offset=True 表示叠加在默认关节位置上
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",  # 动作作用于robot实体
      actuator_names=(".*",),  # 正则匹配所有执行器（所有关节）
      scale=0.25,  # 网络输出×0.25 rad 后叠加到默认关节位置；部分机器人覆盖（如 G1/H1_2/R1），Go2 沿用此值
      use_default_offset=True,  # 在默认关节位置上叠加动作偏移，而非绝对位置控制
    )
  }

  ##
  # 命令：均匀速度采样
  # 每3~8秒重新采样一次速度指令
  # 5%的环境为静止环境（命令为零）
  # heading_command=True：用航向误差生成 ωz（rel_heading_envs 默认 1，非站立环境几乎全开）
  ##

  commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",  # 命令作用于robot实体
      resampling_time_range=(3.0, 8.0),  # 命令重采样间隔（秒），均匀采样3~8s后换新速度指令
      rel_standing_envs=0.05,  # 5%的环境为静止环境（命令为零），帮助学习静止站立
      heading_command=True,  # 启用航向控制模式：用目标航向角误差生成ωz，而非直接命令ωz
      heading_control_stiffness=0.5,  # 航向增益：ωz = clip(stiffness × heading_error, ang_vel_z 范围)
      debug_vis=True,  # 在仿真可视化中绘制命令箭头
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 2.0),  # 前向速度范围 m/s，后退-1~前进+2
        lin_vel_y=(-1.0, 1.0),  # 横向速度 m/s（机体系 y 左为正）：右移-1~左移+1
        ang_vel_z=(-1.0, 1.0),  # 偏航角速度 rad/s（z 上为正）：右转-1~左转+1
        heading=(-math.pi, math.pi),  # 航向角范围 rad，全方向
      ),
    )
  }

  ##
  # 事件：重置和域随机化（Domain Randomization）
  # - reset: episode重置时触发
  # - interval: 训练中周期性触发
  # - startup: 仅在环境创建时触发一次
  ##

  events = {
    "reset_base": EventTermCfg(  # 重置基座位姿（随机xy位置和yaw角）
      func=mdp.reset_root_state_uniform,  # 源码在 mjlab.envs.mdp，经本仓库 mdp 再导出 | 均匀随机重置根节点状态
      mode="reset",  # 在episode重置时触发
      params={
        "pose_range": {  # 位姿随机范围
          "x": (-0.5, 0.5),  # x位置±0.5m
          "y": (-0.5, 0.5),  # y位置±0.5m
          "z": (0.0, 0.0),  # z位置不变（保持地形表面高度）
          "yaw": (-3.14, 3.14),  # 偏航角全范围随机
        },
        "velocity_range": {},  # 速度不随机化（重置为零）
      },
    ),
    "reset_robot_joints": EventTermCfg(  # 重置关节到默认位置
      func=mdp.reset_joints_by_offset,  # 源码在 mjlab.envs.mdp，经本仓库 mdp 再导出 | 按偏移量重置关节位置和速度
      mode="reset",  # 在episode重置时触发
      params={
        "position_range": (-0.0, 0.0),  # 位置偏移为零（重置到精确默认位置）
        "velocity_range": (-0.0, 0.0),  # 速度偏移为零（重置到静止）
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),  # 匹配所有关节
      },
    ),
    "push_robot": EventTermCfg(  # 周期性随机推力扰动（增强鲁棒性）
      func=mdp.push_by_setting_velocity,  # 源码在 mjlab.envs.mdp，经本仓库 mdp 再导出 | 通过直接设置速度施加推力
      mode="interval",  # 训练中周期性触发
      interval_range_s=(5.0, 6.0),  # 每5~6秒推一次
      params={
        "velocity_range": {  # 扰动速度范围
          "x": (-0.5, 0.5),  # x方向±0.5 m/s
          "y": (-0.5, 0.5),  # y方向±0.5 m/s
          "z": (-0.4, 0.4),  # z方向±0.4 m/s
          "roll": (-0.52, 0.52),  # roll角速度±0.52 rad/s（约±30°/s）
          "pitch": (-0.52, 0.52),  # pitch角速度±0.52 rad/s
          "yaw": (-0.78, 0.78),  # yaw角速度±0.78 rad/s（约±45°/s）
        },
      },
    ),
    "foot_friction": EventTermCfg(  # 足底摩擦系数随机化（sim2real关键）
      mode="startup",  # 仅在环境创建时触发一次
      func=dr.geom_friction,  # 源码在 mjlab.envs.mdp.dr | 随机化几何体摩擦系数
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot. 各机器人覆盖geom_names
        "operation": "abs",  # 绝对设置模式（替换原值，而非叠加）
        "ranges": (0.3, 1.6),  # 摩擦系数随机范围[0.3, 1.6]，覆盖默认值
        "shared_random": True,  # All foot geoms share the same friction. 所有足部geom共享同一摩擦系数
      },
    ),
    "encoder_bias": EventTermCfg(  # 编码器偏置随机化（模拟关节编码器误差）
      mode="startup",  # 仅在环境创建时触发一次
      func=dr.encoder_bias,  # 源码在 mjlab.envs.mdp.dr | 随机化关节位置编码器偏置
      params={
        "asset_cfg": SceneEntityCfg("robot"),  # 作用于robot所有关节
        "bias_range": (-0.015, 0.015),  # 偏置范围±0.015 rad（约±0.86°）
      },
    ),
    "base_com": EventTermCfg(  # 基座质心偏移随机化（模拟负载变化）
      mode="startup",  # 仅在环境创建时触发一次
      func=dr.body_com_offset,  # 源码在 mjlab.envs.mdp.dr | 随机化body的质心位置偏移
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot. 各机器人覆盖body_names
        "operation": "add",  # 叠加模式（在原质心上加偏移）
        "ranges": {  # 各轴质心偏移范围
          0: (-0.05, 0.05),  # x轴±0.05m
          1: (-0.05, 0.05),  # y轴±0.05m
          2: (-0.05, 0.05),  # z轴±0.05m
        },
      },
    ),
  }

  ##
  # 奖励：15项，正权重=奖励，负权重=惩罚
  # 跟踪(2) + 姿态/步态(3: orientation/pose/gait) + 平滑(4: ang_vel/angmom/acc/action_rate)
  # + 关节限位(1) + 足部(3) + 终止(1) + 静止(1)
  ##

  rewards = {
    "track_linear_velocity": RewardTermCfg(  # 【+1.0】跟踪机体系 xy 线速度，并加重惩罚 v_z
      func=mdp.track_linear_velocity,  # 源码在 src/tasks/velocity/mdp/rewards.py | exp(-( ||cmd_xy-v_xy||² + 2*v_z² ) / std²)
      weight=1.0,  # 正权重→奖励，鼓励跟踪命令速度
      params={"command_name": "twist", "std": math.sqrt(0.25)},  # std=0.5，控制高斯核宽度，越小对误差越敏感
    ),
    "track_angular_velocity": RewardTermCfg(  # 【+1.0】跟踪命令 ωz（航向模式也已换成角速度）
      func=mdp.track_angular_velocity,  # 源码在 src/tasks/velocity/mdp/rewards.py | 奖励=exp(-||ωz_actual - ωz_cmd||² / std²)
      weight=1.0,  # 正权重→奖励，鼓励跟踪命令角速度
      params={"command_name": "twist", "std": math.sqrt(0.5)},  # std≈0.707，比线速度更宽容
    ),
    "body_orientation_l2": RewardTermCfg(  # 【-1.0】惩罚机身倾斜（保持直立）
      func=mdp.body_orientation_l2,  # 源码在 src/tasks/velocity/mdp/rewards.py | 惩罚=||投影重力偏差||²，偏离直立越大惩罚越重
      weight=-1.0,  # 负权重→惩罚
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot. 各机器人覆盖body_names
    ),
    "pose": RewardTermCfg(  # 【+1.0】默认姿态偏差奖励，站/走/跑三档 std
      func=mdp.variable_posture,  # 源码在 src/tasks/velocity/mdp/rewards.py | exp(-mean(error²/std²))，std 按关节、按档位
      weight=1.0,  # 正权重→奖励，鼓励关节接近默认姿态
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),  # 匹配所有关节
        "command_name": "twist",  # 档位看 ||v_xy||+|ωz|
        "std_standing": {},  # 各机器人覆盖：静止时各关节的std，越小越严格保持默认姿态
        "std_walking": {},  # 各机器人覆盖：行走时各关节的std，允许更大偏差
        "std_running": {},  # 各机器人覆盖：奔跑时各关节的std，允许最大偏差
        "walking_threshold": 0.1,  # ||v_xy||+|ωz| 超过此值进入行走档
        "running_threshold": 1.5,  # ||v_xy||+|ωz| 超过此值进入奔跑档
      },
    ),
    "body_ang_vel": RewardTermCfg(  # 【-0.05】惩罚机身roll/pitch角速度过大
      func=mdp.body_angular_velocity_penalty,  # 源码在 src/tasks/velocity/mdp/rewards.py | 惩罚=||ω_roll, ω_pitch||²
      weight=-0.05,  # 负权重→惩罚，权重较小作为软约束
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot. 各机器人覆盖body_names
    ),
    "angular_momentum": RewardTermCfg(  # 【-0.025】惩罚全身角动量（抑制多余扭转）
      func=mdp.angular_momentum_penalty,  # 源码在 src/tasks/velocity/mdp/rewards.py | 惩罚=||L_total||²，全身绕质心的角动量
      weight=-0.025,  # 负权重→惩罚，权重很小作为软约束
      params={"sensor_name": "robot/root_angmom"},  # 读取robot根节点角动量传感器
    ),
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),  # 源码在 mjlab.envs.mdp，经本仓库 mdp 再导出 | 【-200】提前终止重罚，避免策略学"自杀"行为
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),  # 源码在 mjlab.envs.mdp，经本仓库 mdp 再导出 | 【-2.5e-7】关节加速度L2惩罚，鼓励平滑加速，权重极小作为正则项
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),  # 源码在 mjlab.envs.mdp，经本仓库 mdp 再导出 | 【-10】越出软限位的量再惩罚，防止关节超限
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),  # 源码在 mjlab.envs.mdp，经本仓库 mdp 再导出 | 【-0.05】动作变化率L2惩罚，||a_t - a_{t-1}||²，鼓励平滑控制
    "foot_gait": RewardTermCfg(  # 【+0.5】步态节奏奖励（对角步态）
      func=mdp.feet_gait,  # 源码在 src/tasks/velocity/mdp/rewards.py | 奖励足在正确相位触地，对角足交替支撑
      weight=0.5,  # 正权重→奖励
      params={
        "period": 0.6,  # 步态周期0.6s
        "offset": [0.0, 0.5],  # 占位；四足须覆盖为每腿一项（Go2 FR,FL,RR,RL → [0,0.5,0.5,0]）
        "threshold": 0.56,  # 期望支撑相占周期的比例：leg_phase < 0.56 应触地，否则应腾空
        "command_threshold": 0.1,  # ||v_xy||+|ωz| 低于此值不给步态奖励
        "command_name": "twist",  # 引用速度命令
        "sensor_name": "feet_ground_contact",  # 引用足部接触传感器
      }
    ),
    "foot_clearance": RewardTermCfg(  # 【-1.0】|h-h_target|×||v_xy||（过高过低都罚，乘足端水平速度）
      func=mdp.feet_clearance,  # 源码在 src/tasks/velocity/mdp/rewards.py | 对所有足求和，不区分摆动/支撑
      weight=-1.0,  # 负权重→惩罚
      params={
        "target_height": 0.10,  # 目标足端世界系高度 0.1m
        "command_name": "twist",  # 引用速度命令
        "command_threshold": 0.1,  # ||v_xy||+|ωz| 低于此值不施加
        "asset_cfg": SceneEntityCfg("robot", site_names=())},  # Set per-robot. 各机器人覆盖site_names
    ),
    "foot_slip": RewardTermCfg(  # 【-0.25】足底滑行：触地足 ||v_xy||² 求和
      func=mdp.feet_slip,  # 源码在 src/tasks/velocity/mdp/rewards.py | 只对触地足计水平速度平方
      weight=-0.25,  # 负权重→惩罚
      params={
        "sensor_name": "feet_ground_contact",  # 引用足部接触传感器判断是否触地
        "command_name": "twist",  # 引用速度命令
        "command_threshold": 0.1,  # ||v_xy||+|ωz| 低于此值不施加
        "asset_cfg": SceneEntityCfg("robot", site_names=())},  # Set per-robot. 各机器人覆盖site_names
    ),
    "soft_landing": RewardTermCfg(  # 【-1e-3】着地冲击：首次触地时 ||f|| 求和
      func=mdp.soft_landing,  # 源码在 src/tasks/velocity/mdp/rewards.py | 三维接触力模长，不是单独的 z 向力
      weight=-1e-3,  # 负权重→惩罚，权重很小作为软约束
      params={
        "sensor_name": "feet_ground_contact",  # 引用足部接触传感器
        "command_name": "twist",  # 引用速度命令
        "command_threshold": 0.1,  # ||v_xy||+|ωz| 低于此值不施加
      },
    ),
    "stand_still": RewardTermCfg(  # 【-1.0】低速命令时关节偏离默认位置
      func=mdp.stand_still,  # 源码在 src/tasks/velocity/mdp/rewards.py | ||v_xy||+|ωz|≤阈值 时 Σ(q-q_default)²
      weight=-1.0,  # 负权重→惩罚
      params={
        "command_name": "twist",  # 引用速度命令
        "command_threshold": 0.1,  # ||v_xy||+|ωz| 低于此值视为静止
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),  # 匹配所有关节
      },
    ),
  }

  ##
  # 终止条件
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),  # 源码在 mjlab.envs.mdp，经本仓库 mdp 再导出 | episode超时（20s），time_out=True标记为超时而非失败
    "fell_over": TerminationTermCfg(  # 机身倾倒超过70°则终止
      func=mdp.bad_orientation,  # 源码在 mjlab.envs.mdp，经本仓库 mdp 再导出 | 检测机身姿态角是否超过限制
      params={"limit_angle": math.radians(70.0)},  # 极限角度70°≈1.22 rad，超过则判定倾倒
    ),
  }

  ##
  # 课程学习：逐步增加训练难度
  ##

  curriculum = {
    "terrain_levels": CurriculumTermCfg(  # 地形难度渐进：走得远→升级到更难地形
      func=mdp.terrain_levels_vel,  # 源码在 src/tasks/velocity/mdp/curriculums.py | 根据机器人移动距离自动提升地形难度等级
      params={"command_name": "twist"},  # 用命令算「该走多远」门槛，再按实际走距升降地形等级
    ),
    "command_vel": CurriculumTermCfg(  # 命令速度渐进：计数器超过各 stage.step 后改 ranges
      func=mdp.commands_vel,  # 源码在 src/tasks/velocity/mdp/curriculums.py | common_step_counter > step 时写入范围
      params={
        "command_name": "twist",  # 引用速度命令
        "velocity_stages": [  # 注意：commands 出厂已是大范围；>0 后才收到本档
          {"step": 0, "lin_vel_x": (-0.5, 1.0), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.0, 1.0)},  # 第一步之后收到小范围
          {"step": 5000 * 24, "lin_vel_x": (-1.0, 2.0), "lin_vel_y": (-1.0, 1.0)},  # 12万 env step 后扩大 xy（ωz 沿用上档）
        ],
      },
    ),
  }

  ##
  # 组装并返回完整的环境配置
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(  # 默认使用地形生成器（粗糙地形），平坦地形会覆盖此设置
        terrain_type="generator",  # 地形类型：generator=程序化生成粗糙地形，plane=平坦地面
        terrain_generator=replace(ROUGH_TERRAINS_CFG),  # 粗糙地形生成器配置（含多种地形类型和课程参数）
        max_init_terrain_level=5,  # 初始地形难度最大等级（0~5），课程学习会逐步提升
      ),
      sensors=(terrain_scan,),  # 场景传感器列表，仅包含地形扫描raycast传感器
      num_envs=1,  # 默认1个环境，训练时通过CLI覆盖（如--env.scene.num-envs=4096）
      extent=2.0,  # 多环境布置间距相关（mjlab SceneCfg），不是把机器人裁出世界
    ),
    observations=observations,  # 观测配置（actor+critic两组）
    actions=actions,  # 动作配置（关节位置控制）
    commands=commands,  # 命令配置（均匀速度采样）
    events=events,  # 事件配置（重置+域随机化）
    rewards=rewards,  # 奖励配置（15项）
    terminations=terminations,  # 终止条件（超时+倾倒）
    curriculum=curriculum,  # 课程学习（地形+速度渐进）
    metrics=metrics,  # 评估指标（动作平滑度）
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,  # 相机跟随模式：跟随指定body
      entity_name="robot",  # 跟随robot实体
      body_name="",  # 各机器人覆盖，指定跟随的具体body名称
      distance=3.0,  # 相机距离目标的距离3m
      elevation=-5.0,  # 相机仰角-5°（略俯视）
      azimuth=90.0,  # 相机方位角90°（侧面视角）
    ),
    sim=SimulationCfg(
      nconmax=35,  # 最大接触数，MuJoCo接触缓冲区大小
      njmax=1500,  # 最大约束Jacobian行数，MuJoCo约束缓冲区大小
      mujoco=MujocoCfg(
        timestep=0.005,  # 仿真步长5ms（200Hz物理仿真）
        iterations=10,  # 求解器迭代次数，影响约束求解精度
        ls_iterations=20,  # 线搜索迭代次数，影响求解稳定性
      ),
    ),
    decimation=4,  # 仿真4步输出1次 → 控制频率 = 1/(0.005*4) = 50Hz
    episode_length_s=20.0,  # 每个episode 20秒
  )