// Go2部署入口：C++ FSM状态机控制真实机器人
// 状态转换：Passive → FixStand → RLBase
// 操作：按[L2+up]进入FixStand，按[R2+A]启动RL控制
#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;

void init_fsm_state()
{
    // 检查是否有其他进程占用lowcmd通道
    auto lowcmd_sub = std::make_shared<unitree::robot::go2::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
        // exit(0);
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

int main(int argc, char** argv)
{
    // Load parameters
    auto vm = param::helper(argc, argv);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     Go2 Controller \n";

    // Unitree DDS通信初始化（与机器人低层通信）
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    init_fsm_state();

    // 初始化FSM状态机（从配置文件加载状态转换规则）
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    std::cout << "Press [L2 + up] to enter FixStand mode.\n";
    std::cout << "And then press [R2 + A] to start controlling the robot.\n";

    while (true)
    {
        sleep(1);
    }
    
    return 0;
}