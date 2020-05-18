###############################################
# 线下仿真运行一键启动脚本
# 以下顺序可以不分先后，不需要的功能可以屏蔽
###############################################

export  CYBER_PATH=/home/watrix18/workspace/AiwSys/exec/cyber
# 程序日志目录
export  GLOG_log_dir=/home/watrix18/workspace/adas_data/logs
#ADAS 配置目录
export ADAS_PATH=/home/watrix18/workspace/AiwSys/exec/projects/adas

# 启动仿真测试
#启动前配置好exec/projects/adas/production/conf/adas_simulator_config.pb.txt
eval python3 cyber_launch start /home/watrix18/workspace/AiwSys/exec/projects/adas/production/launch/simulator.launch &

# 启动感知功能
#启动前配置好exec/projects/adas/production/conf/adas_perception_config.pb.txt
eval python3 cyber_launch start /home/watrix18/workspace/AiwSys/exec/projects/adas/production/launch/perception.launch &
# 启动回放/录像功能
# 记得配置exec/projects/adas/production/conf/adas_recorder_config.pb.txt
eval python3 cyber_launch start /home/watrix18/workspace/AiwSys/exec/projects/adas/production/launch/recorder.launch &
#启动运行HMI
eval /home/watrix18/workspace/AiwSys/bazel-bin/projects/adas/component/hmi/perception_hmi &
