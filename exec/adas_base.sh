
# 获得脚本所在的绝对路径
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

ROOT_DIR="$( cd "${DIR}/.." && pwd )" 

#设置工作目录变量
export ADAS_WORK_PATH=${DIR}

# 程序bin目录 ./cyber_lauch 脚本要用
export  ADAS_WORK_BIN=${ROOT_DIR}/bazel-bin

#CYBER配置目录
export  CYBER_PATH=${ADAS_WORK_PATH}/cyber
#CYBER_IP  配置這個IP可以启用RTPS， 实现局域网通信
export  CYBER_IP=192.168.0.107
#ADAS 配置目录
export ADAS_CONFIG_PATH=${ADAS_WORK_PATH}/projects/adas

# 程序日志目录
export  GLOG_log_dir=${ADAS_WORK_PATH}/logs

echo "ROOT_DIR:${ROOT_DIR}"
echo "ADAS_WORK_PATH:${ADAS_WORK_PATH}"
echo "ADAS_WORK_BIN:${ADAS_WORK_BIN}"
echo "ADAS_CONFIG_PATH:${ADAS_CONFIG_PATH}"
echo "GLOG_log_dir:${GLOG_log_dir}"


if [ -d ${GLOG_log_dir} ]; then
echo "GLOG目录存在:${GLOG_log_dir}"
else
echo "GLOG目录不存在 , 创建:${GLOG_log_dir}"
mkdir -p ${GLOG_log_dir}
fi

