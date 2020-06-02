###############################################
# 线下仿真运行一键启动脚本
# 以下顺序可以不分先后，不需要的功能可以屏蔽
###############################################
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "${DIR}/.."

source "${DIR}/adas_base.sh"

# 启动仿真测试
#启动前配置好exec/projects/adas/production/conf/adas_simulator_config.pb.txt
eval python3 ${ADAS_WORK_PATH}/cyber_launch start ${ADAS_WORK_PATH}/projects/adas/production/launch/simulator.launch &

# 启动感知功能
#启动前配置好exec/projects/adas/production/conf/adas_perception_config.pb.txt
eval python3 ${ADAS_WORK_PATH}/cyber_launch start ${ADAS_WORK_PATH}/projects/adas/production/launch/perception.launch &
# 启动回放/录像功能
# 记得配置exec/projects/adas/production/conf/adas_recorder_config.pb.txt
eval python3 ${ADAS_WORK_PATH}/cyber_launch start ${ADAS_WORK_PATH}/projects/adas/production/launch/recorder.launch &

# 启动HMI功能
eval python3 ${ADAS_WORK_PATH}/cyber_launch start ${ADAS_WORK_PATH}/projects/adas/production/launch/hmi.launch &
