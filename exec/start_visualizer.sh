###############################################
#独立启动HMI
###############################################
export  CYBER_PATH=/media/shuimujie/C14D581BDA18EBFA/10.Projects/01.Linux/02.github/AiwSys/cyber
# 程序日志目录
export  GLOG_log_dir=/media/shuimujie/C14D581BDA18EBFA/10.Projects/01.Linux/adas_data/logs
#ADAS 配置目录
export ADAS_PATH=/media/shuimujie/C14D581BDA18EBFA/10.Projects/01.Linux/02.github/AiwSys/projects/adas
#设置工作目录变量
ADAS_WORK_PATH=/media/shuimujie/C14D581BDA18EBFA/10.Projects/01.Linux/02.github/AiwSys

echo "work path : ${ADAS_WORK_PATH}"
#启动运行HMI
eval ${ADAS_WORK_PATH}/bazel-bin/modules/tools/visualizer/cyber_visualizer &