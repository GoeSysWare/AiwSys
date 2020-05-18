###############################################
#独立启动感知
###############################################
export  CYBER_PATH=/home/watrix18/workspace/AiwSys/exec/cyber
# 程序日志目录
export  GLOG_log_dir=/home/watrix18/workspace/adas_data/logs
#ADAS 配置目录
export ADAS_PATH=/home/watrix18/workspace/AiwSys/exec/projects/adas


# 启动感知功能
#启动前配置好exec/projects/adas/production/conf/adas_perception_config.pb.txt
python3 cyber_launch start /home/watrix18/workspace/AiwSys/exec/projects/adas/production/launch/perception.launch

