###############################################
# 线上运行一键启动脚本
# 以下顺序可以不分先后，不需要的功能可以屏蔽
###############################################
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "${DIR}/.."

source "${DIR}/adas_base.sh"


# 启动照相机
#启动前配置好/media/shuimujie/C14D581BDA18EBFA/10.Projects/01.Linux/02.github/AiwSys//exec/modules/drivers/camera/conf/camera_front_6mm.pb.txt
eval python3 ${ADAS_WORK_PATH}/cyber_launch start ${ADAS_WORK_PATH}/modules/drivers/camera/launch/camera.launch &
sleep 1

# 启动velodyne雷达
#eval python3 ${ADAS_WORK_PATH}/cyber_launch start ${ADAS_WORK_PATH}/modules/drivers/velodyne/launch/velodyne.launch &
# sleep 1

# 启动Innovision雷达
eval python3 ${ADAS_WORK_PATH}/cyber_launch start ${ADAS_WORK_PATH}/modules/drivers/innovision/launch/innovision.launch &
sleep 1

# 启动感知功能
#启动前配置好exec/projects/adas/production/conf/adas_perception_config.pb.txt
eval python3 ${ADAS_WORK_PATH}/cyber_launch start ${ADAS_WORK_PATH}/projects/adas/production/launch/perception.launch &
sleep 5
# 启动回放/录像功能
# 记得配置exec/projects/adas/production/conf/adas_recorder_config.pb.txt
eval python3 ${ADAS_WORK_PATH}/cyber_launch start ${ADAS_WORK_PATH}/projects/adas/production/launch/recorder.launch &
sleep 1
# 启动HMI功能
eval python3 ${ADAS_WORK_PATH}/cyber_launch start ${ADAS_WORK_PATH}/projects/adas/production/launch/hmi.launch &
