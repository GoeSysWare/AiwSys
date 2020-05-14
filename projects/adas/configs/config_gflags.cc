/******************************************************************************
 * Copyright 2017 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

#include "projects/adas/configs/config_gflags.h"



DEFINE_string(adas_cfg_interface_file,
"production/conf/adas_interface.pb.txt",
"ADAS系统内置参数配置文件，如果无此文件，系统自动采用默认值");

DEFINE_string(adas_cfg_algorithm_file,
"production/conf/adas_algorithm.pb.txt",
"算法模块级的参数配置文件");



DEFINE_string(adas_debug_output_dir,
"data/debug/",
"ADAS系统调试保存Images的输出目录");


DEFINE_int32(adas_camera_size,2,"ADAS支持的相机数");
DEFINE_int32(adas_lidar_size,1,"ADAS支持的激光雷达数");


DEFINE_string(distance_cfg_long_a,
"production/conf/autotrain_models/distance_table/long_a.bin",
"配置文件");
DEFINE_string(distance_cfg_long_b,
"production/conf/autotrain_models/distance_table/long_b.bin",
"配置文件");
DEFINE_string(distance_cfg_short_a,
"production/conf/autotrain_models/distance_table/short_a.bin",
"配置文件");
DEFINE_string(distance_cfg_short_b,
"production/conf/autotrain_models/distance_table/short_b.bin",
"配置文件");

DEFINE_string(calibrator_cfg_short,
"production/conf/nodes/config/camera_short.yaml",
"配置文件");
DEFINE_string(calibrator_cfg_long,
"production/conf/nodes/config/camera_long.yaml",
"配置文件");

DEFINE_string(calibrator_cfg_distortTable,
"production/conf/nodes/config/distortTable.bin",
"配置文件");

DEFINE_string(lidar_map_parameter,
"production/conf/nodes/config/lidar_map_image.yaml",
"配置文件");

