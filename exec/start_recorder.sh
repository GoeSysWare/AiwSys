###############################################
#独立启动回放/录像
###############################################
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "${DIR}/.."

source "${DIR}/adas_base.sh"

# 启动回放/录像功能
# 记得配置 projects/adas/production/conf/adas_recorder_config.pb.txt
eval python3 ${ADAS_WORK_PATH}/cyber_launch start ${ADAS_WORK_PATH}/projects/adas/production/launch/recorder.launch &

