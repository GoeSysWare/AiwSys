###############################################
#独立启动lidar
###############################################
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "${DIR}/.."

source "${DIR}/adas_base.sh"
# 启动velodyne雷达
#eval python3 ${ADAS_WORK_PATH}/cyber_launch start ${ADAS_WORK_PATH}/modules/drivers/velodyne/launch/velodyne.launch &


# 启动Innovision雷达
eval python3 ${ADAS_WORK_PATH}/cyber_launch start ${ADAS_WORK_PATH}/modules/drivers/innovision/launch/innovision.launch &