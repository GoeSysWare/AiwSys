###############################################
# 线上运行一键停止脚本
# 以下顺序可以不分先后，不需要的功能可以屏蔽
###############################################
export  CYBER_PATH=/home/watrix18/workspace/AiwSys/exec/cyber
# 程序日志目录
export  GLOG_log_dir=/home/watrix18/workspace/adas_data/logs
#ADAS 配置目录
export ADAS_PATH=/home/watrix18/workspace/AiwSys/exec/projects/adas

# 停止照相机
#启动前配置好/home/watrix18/workspace/AiwSys/exec/modules/drivers/camera/conf/camera_front_6mm.pb.txt
eval python3 cyber_launch stop /home/watrix18/workspace/AiwSys/exec/modules/drivers/camera/launch/camera.launch &

# 停止感知功能
#启动前配置好exec/projects/adas/production/conf/adas_perception_config.pb.txt
eval python3 cyber_launch stop /home/watrix18/workspace/AiwSys/exec/projects/adas/production/launch/perception.launch &

#停止运行HMI
eval kill -9 $(ps -ef |grep perception_hmi)





