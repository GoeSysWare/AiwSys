###############################################
#独立启动可视化调试工具
###############################################
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "${DIR}/.."

source "${DIR}/adas_base.sh"


#启动运行HMI
eval ${ADAS_WORK_BIN}/modules/tools/visualizer/cyber_visualizer &