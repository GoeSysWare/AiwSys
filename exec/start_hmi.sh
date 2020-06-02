###############################################
#独立启动HMI
###############################################
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "${DIR}/.."

source "${DIR}/adas_base.sh"

# 启动HMI功能

eval python3 ${ADAS_WORK_PATH}/cyber_launch start ${ADAS_WORK_PATH}/projects/adas/production/launch/hmi.launch &
