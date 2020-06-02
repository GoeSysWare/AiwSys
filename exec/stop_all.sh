###############################################
# 清理所有的进程
# 以下顺序可以不分先后，不需要的功能可以屏蔽
###############################################
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "${DIR}/.."

source "${DIR}/adas_base.sh"

# 停止回放/录像功能
# 记得配置  projects/adas/production/conf/adas_recorder_config.pb.txt
eval python3 ${ADAS_WORK_PATH}/cyber_launch stop ${ADAS_WORK_PATH}/projects/adas/production/launch/recorder.launch &
sleep 1

# 停止感知功能
#停止前配置好 projects/adas/production/conf/adas_perception_config.pb.txt
eval python3 ${ADAS_WORK_PATH}/cyber_launch stop ${ADAS_WORK_PATH}/projects/adas/production/launch/perception.launch &
sleep 5

sleep 1
# 停止雷达
# eval python3 ${ADAS_WORK_PATH}/cyber_launch stop ${ADAS_WORK_PATH}/modules/drivers/velodyne/launch/velodyne.launch &
# sleep 1
# 停止Innovision雷达
eval python3 ${ADAS_WORK_PATH}/cyber_launch stop ${ADAS_WORK_PATH}/modules/drivers/innovision/launch/innovision.launch &
sleep 1
# 停止照相机
eval python3 ${ADAS_WORK_PATH}/cyber_launch stop ${ADAS_WORK_PATH}/modules/drivers/camera/launch/camera.launch &
sleep 1

# 停止仿真测试
eval python3 ${ADAS_WORK_PATH}/cyber_launch stop ${ADAS_WORK_PATH}/projects/adas/production/launch/simulator.launch &
sleep 1

# 启动WebView功能
# eval python3 ${ADAS_WORK_PATH}/cyber_launch stop ${ADAS_WORK_PATH}/projects/adas/production/launch/webview.launch  &
sleep 1

# 停止HMI功能
eval python3 ${ADAS_WORK_PATH}/cyber_launch stop ${ADAS_WORK_PATH}/projects/adas/production/launch/hmi.launch &
sleep 1




