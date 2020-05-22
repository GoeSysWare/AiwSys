###############################################
# 线上运行一键停止脚本
# 以下顺序可以不分先后，不需要的功能可以屏蔽
###############################################
export  CYBER_PATH=/media/shuimujie/C14D581BDA18EBFA/10.Projects/01.Linux/02.github/AiwSys/cyber
# 程序日志目录
export  GLOG_log_dir=/media/shuimujie/C14D581BDA18EBFA/10.Projects/01.Linux/adas_data/logs
#ADAS 配置目录
export ADAS_PATH=/media/shuimujie/C14D581BDA18EBFA/10.Projects/01.Linux/02.github/AiwSys/projects/adas


#设置工作目录变量
ADAS_WORK_PATH=/media/shuimujie/C14D581BDA18EBFA/10.Projects/01.Linux/02.github/AiwSys

echo "work path : ${ADAS_WORK_PATH}"
# 停止照相机
#启动前配置好 modules/drivers/camera/conf/camera_front_6mm.pb.txt
eval python3 cyber_launch stop ${ADAS_WORK_PATH}/modules/drivers/camera/launch/camera.launch &

# 停止感知功能
#启动前配置好 projects/adas/production/conf/adas_perception_config.pb.txt
eval python3 cyber_launch stop ${ADAS_WORK_PATH}/projects/adas/production/launch/perception.launch &

#停止运行HMI
# eval python3 cyber_launch stop ${ADAS_WORK_PATH}/projects/adas/production/launch/hmi.launch &





