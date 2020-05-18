###############################################
#独立启动仿真
###############################################
export  CYBER_PATH=/home/watrix18/workspace/AiwSys/exec/cyber
# 程序日志目录
export  GLOG_log_dir=/home/watrix18/workspace/adas_data/logs
#ADAS 配置目录
export ADAS_PATH=/home/watrix18/workspace/AiwSys/exec/projects/adas


# 启动仿真测试
#启动前配置好exec/projects/adas/production/conf/adas_simulator_config.pb.txt
python3 cyber_launch start /home/watrix18/workspace/AiwSys/exec/projects/adas/production/launch/simulator.launch

