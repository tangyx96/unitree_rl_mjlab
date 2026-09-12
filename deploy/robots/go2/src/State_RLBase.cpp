// RL 推理状态：加载 ONNX 与 deploy.yaml，把策略输出的关节位置写入 lowcmd。
// 观测构建与 ONNX 推理由 ManagerBasedRLEnv 的 step（通常在 FSM 周期里调用）完成；
// 本 run() 只读取 processed_actions() 并下发电机目标。
#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    // 创建部署环境：加载deploy.yaml配置 + 绑定真实机器人状态
    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    // 加载ONNX策略模型
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    // 注册安全检查：机身倾倒时自动切换到Passive状态
    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    // 读取本控制周期已处理好的关节目标（step 在 FSM 其它处调用）
    auto action = env->action_manager->processed_actions();
    // 将动作（关节位置命令）写入lowcmd，通过DDS发送给机器人
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}