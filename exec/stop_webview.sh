###############################################
#独立停止WebView
###############################################
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "${DIR}/.."

source "${DIR}/adas_base.sh"


# 停止WebView功能
eval python3 ${ADAS_WORK_PATH}/cyber_launch stop ${ADAS_WORK_PATH}/projects/adas/production/launch/webview.launch  &
