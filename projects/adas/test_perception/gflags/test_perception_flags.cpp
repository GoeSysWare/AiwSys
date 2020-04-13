#include "test_perception_flags.h"


DEFINE_string(test_config_filepath,
	"/media/shuimujie/C14D581BDA18EBFA/10.Projects/01.Linux/02.github/AiwSys/projects/adas/cfg/nodes/config/test_config.pb.txt",
	"The test config filename");
	
	DEFINE_string(node_config_filepath,
	"/media/shuimujie/C14D581BDA18EBFA/10.Projects/01.Linux/02.github/AiwSys/projects/adas/cfg/nodes/config/node_config.pb.txt",
	"The node config filename");

	DEFINE_string(perception_adapter_filepath,
	"/media/shuimujie/C14D581BDA18EBFA/10.Projects/01.Linux/02.github/AiwSys/projects/adas/cfg/nodes/adapter/perception.conf",
	"The adapter config filename");


DEFINE_string(distance_cfg_long_a,
"/media/shuimujie/C14D581BDA18EBFA/10.Projects/01.Linux/02.github/AiwSys/projects/adas/cfg/autotrain_models/distance_table/long_a.bin",
"配置文件");
DEFINE_string(distance_cfg_long_b,
"/media/shuimujie/C14D581BDA18EBFA/10.Projects/01.Linux/02.github/AiwSys/projects/adas/cfg/autotrain_models/distance_table/long_b.bin",
"配置文件");
DEFINE_string(distance_cfg_short_a,
"/media/shuimujie/C14D581BDA18EBFA/10.Projects/01.Linux/02.github/AiwSys/projects/adas/cfg/autotrain_models/distance_table/short_a.bin",
"配置文件");
DEFINE_string(distance_cfg_short_b,
"/media/shuimujie/C14D581BDA18EBFA/10.Projects/01.Linux/02.github/AiwSys/projects/adas/cfg/autotrain_models/distance_table/short_b.bin",
"配置文件");


DEFINE_string(calibrator_cfg_short,
"/media/shuimujie/C14D581BDA18EBFA/10.Projects/01.Linux/02.github/AiwSys/projects/adas/cfg/nodes/config/camera_short.yaml",
"配置文件");
DEFINE_string(calibrator_cfg_long,
"/media/shuimujie/C14D581BDA18EBFA/10.Projects/01.Linux/02.github/AiwSys/projects/adas/cfg/nodes/config/camera_long.yaml",
"配置文件");


DEFINE_string(calibrator_cfg_distortTable,
"/media/shuimujie/C14D581BDA18EBFA/10.Projects/01.Linux/02.github/AiwSys/projects/adas/cfg/nodes/config/distortTable.bin",
"配置文件");


DEFINE_string(lidar_map_parameter,
"/media/shuimujie/C14D581BDA18EBFA/10.Projects/01.Linux/02.github/AiwSys/projects/adas/cfg/nodes/config/lidar_map_image.yaml",
"配置文件");




