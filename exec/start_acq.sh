###############################################
# 线上启动相机检测和保存脚本
# 需要独立配置好production/conf/adas_perception_config.pb.txt
###############################################

export  CYBER_PATH=/media/shuimujie/C14D581BDA18EBFA/10.Projects/01.Linux/02.github/AiwSys/cyber
# 程序日志目录
export  GLOG_log_dir=/media/shuimujie/C14D581BDA18EBFA/10.Projects/01.Linux/adas_data/logs
#ADAS 配置目录
export ADAS_PATH=/media/shuimujie/C14D581BDA18EBFA/10.Projects/01.Linux/02.github/AiwSys/projects/adas

#设置工作变量
ADAS_WORK_PATH=/media/shuimujie/C14D581BDA18EBFA/10.Projects/01.Linux/02.github/AiwSys

echo "work path : ${ADAS_WORK_PATH}"

# 启动照相机
#启动前配置好/home/watrix18/workspace/AiwSys/exec/modules/drivers/camera/conf/camera_front_6mm.pb.txt
eval python3 cyber_launch start ${ADAS_WORK_PATH}/modules/drivers/camera/launch/camera.launch &

# 启动感知功能
#启动前配置好exec/projects/adas/production/conf/adas_perception_config.pb.txt
eval python3 cyber_launch start ${ADAS_WORK_PATH}/projects/adas/production/launch/perception.launch &

#启动运行HMI
eval ${ADAS_WORK_PATH}/bazel-bin/projects/adas/component/hmi/perception_hmi &
