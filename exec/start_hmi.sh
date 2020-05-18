###############################################
#独立启动HMI
###############################################
export  CYBER_PATH=/home/watrix18/workspace/AiwSys/exec/cyber
# 程序日志目录
export  GLOG_log_dir=/home/watrix18/workspace/adas_data/logs
#ADAS 配置目录
export ADAS_PATH=/home/watrix18/workspace/AiwSys/exec/projects/adas

#启动运行HMI
/home/watrix18/workspace/AiwSys/bazel-bin/projects/adas/component/hmi/perception_hmi