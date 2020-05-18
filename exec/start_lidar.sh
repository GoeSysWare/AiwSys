###############################################
#独立启动lidar
###############################################
export  CYBER_PATH=/home/watrix18/workspace/AiwSys/exec/cyber
# 程序日志目录
export  GLOG_log_dir=/home/watrix18/workspace/adas_data/logs
#ADAS 配置目录
export ADAS_PATH=/home/watrix18/workspace/AiwSys/exec/projects/adas

# 启动雷达
python3 cyber_launch start /home/watrix18/workspace/AiwSys/exec/modules/drivers/velodyne/launch/velodyne.launch