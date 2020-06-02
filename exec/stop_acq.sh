###############################################
# 线上运行一键停止脚本
# 以下顺序可以不分先后，不需要的功能可以屏蔽
###############################################
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "${DIR}/.."

source "${DIR}/adas_base.sh"


# 停止照相机
#启动前配置好 modules/drivers/camera/conf/camera_front_6mm.pb.txt
eval python3 ${ADAS_WORK_PATH}/cyber_launch stop ${ADAS_WORK_PATH}/modules/drivers/camera/launch/camera.launch &

# 停止感知功能
#启动前配置好 projects/adas/production/conf/adas_perception_config.pb.txt
eval python3 ${ADAS_WORK_PATH}/cyber_launch stop ${ADAS_WORK_PATH}/projects/adas/production/launch/perception.launch &







