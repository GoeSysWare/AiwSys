# ADAS 仿真运行模块 
该模块是Adas的仿真运行模块，实现离线仿真、测试、演示效果
解析本地的图片(.jpg .png .jpeg等格式)和Lidar(.pcd格式) 
## 使用介绍 
### 配置 
+ **配置文件**: adas_simulator.pb.txt
+ **simulator_files_dir**: 仿真文件所处的本地路径 
+ **config_file**: 仿真文件的匹配表，格式是逗号分隔符的文件，内容为 ''' 相机1文件路径,相机2文件路径,雷达文件路径'''  
**注意**: config_file 文件路径可以为相对路径和绝对路径，当为相对路径时每个文件的总路径为 simulator_files_dir + config_file
 ### 软件启动
 + **独立启动脚本**: /exec/sart_simulator.sh 
 + **脚本配置** : 需要设定各个路径
 ## 注意事项
