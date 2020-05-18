###############################################
# 线上运行一键启动脚本
# 以下顺序可以不分先后，不需要的功能可以屏蔽
###############################################

export  CYBER_PATH=/home/watrix18/workspace/AiwSys/exec/cyber
# 程序日志目录
export  GLOG_log_dir=/home/watrix18/workspace/adas_data/logs
#ADAS 配置目录
export ADAS_PATH=/home/watrix18/workspace/AiwSys/exec/projects/adas


# 启动照相机
#启动前配置好/home/watrix18/workspace/AiwSys/exec/modules/drivers/camera/conf/camera_front_6mm.pb.txt
eval python3 cyber_launch start /home/watrix18/workspace/AiwSys/exec/modules/drivers/camera/launch/camera.launch &
sleep 1
# 启动雷达
eval python3 cyber_launch start /home/watrix18/workspace/AiwSys/exec/modules/drivers/velodyne/launch/velodyne.launch &
sleep 1
# 启动感知功能
#启动前配置好exec/projects/adas/production/conf/adas_perception_config.pb.txt
eval python3 cyber_launch start /home/watrix18/workspace/AiwSys/exec/projects/adas/production/launch/perception.launch &
sleep 5
# 启动回放/录像功能
# 记得配置exec/projects/adas/production/conf/adas_recorder_config.pb.txt
eval python3 cyber_launch start /home/watrix18/workspace/AiwSys/exec/projects/adas/production/launch/recorder.launch &
sleep 1
#启动运行HMI
eval /home/watrix18/workspace/AiwSys/bazel-bin/projects/adas/component/hmi/perception_hmi &

