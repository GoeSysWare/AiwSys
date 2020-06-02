###############################################
# 线上启动相机检测和保存脚本
# 需要独立配置好production/conf/adas_perception_config.pb.txt
###############################################

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "${DIR}/.."

source "${DIR}/adas_base.sh"

# 启动照相机
#启动前配置好/media/shuimujie/C14D581BDA18EBFA/10.Projects/01.Linux/02.github/AiwSys//exec/modules/drivers/camera/conf/camera_front_6mm.pb.txt
eval python3 ${ADAS_WORK_PATH}/cyber_launch start ${ADAS_WORK_PATH}/modules/drivers/camera/launch/camera.launch &

# 启动感知功能
#启动前配置好exec/projects/adas/production/conf/adas_perception_config.pb.txt
eval python3 ${ADAS_WORK_PATH}/cyber_launch start ${ADAS_WORK_PATH}/projects/adas/production/launch/perception.launch &

