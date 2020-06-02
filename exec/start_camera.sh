###############################################
#独立启动相机、lidar
###############################################
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "${DIR}/.."

source "${DIR}/adas_base.sh"

# 启动照相机
eval python3 ${ADAS_WORK_PATH}/cyber_launch start ${ADAS_WORK_PATH}/modules/drivers/camera/launch/camera.launch &

