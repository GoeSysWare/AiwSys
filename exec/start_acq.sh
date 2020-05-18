###############################################
# 线上启动相机检测和保存脚本
# 需要独立配置好production/conf/adas_perception_config.pb.txt
###############################################

export  CYBER_PATH=/home/watrix18/workspace/AiwSys/exec/cyber
# 程序日志目录
export  GLOG_log_dir=/home/watrix18/workspace/adas_data/logs
#ADAS 配置目录
export ADAS_PATH=/home/watrix18/workspace/AiwSys/exec/projects/adas

# 启动照相机
#启动前配置好/home/watrix18/workspace/AiwSys/exec/modules/drivers/camera/conf/camera_front_6mm.pb.txt
eval python3 cyber_launch start /home/watrix18/workspace/AiwSys/exec/modules/drivers/camera/launch/camera.launch &

# 启动感知功能
#启动前配置好exec/projects/adas/production/conf/adas_perception_config.pb.txt
eval python3 cyber_launch start /home/watrix18/workspace/AiwSys/exec/projects/adas/production/launch/perception.launch &

#启动运行HMI
eval /home/watrix18/workspace/AiwSys/bazel-bin/projects/adas/component/hmi/perception_hmi &
