###############################################
#独立启动相机、lidar
###############################################
export  CYBER_PATH=/home/watrix18/workspace/AiwSys/exec/cyber
# 程序日志目录
export  GLOG_log_dir=/home/watrix18/workspace/adas_data/logs
#ADAS 配置目录
export ADAS_PATH=/home/watrix18/workspace/AiwSys/exec/projects/adas


# 启动照相机
python3 cyber_launch start /home/watrix18/workspace/AiwSys/exec/modules/drivers/camera/launch/camera.launch

