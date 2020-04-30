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
"conf/adas_interface.pb.txt",
"ADAS系统内置参数配置文件，如果无此文件，系统自动采用默认值");

DEFINE_int32(adas_camera_size,2,"ADAS支持的相机数");
DEFINE_int32(adas_lidar_size,1,"ADAS支持的激光雷达数");


DEFINE_string(distance_cfg_long_a,
"conf/distance_table/long_a.bin",
"配置文件");
DEFINE_string(distance_cfg_long_b,
"conf/distance_table/long_b.bin",
"配置文件");
DEFINE_string(distance_cfg_short_a,
"conf/distance_table/short_a.bin",
"配置文件");
DEFINE_string(distance_cfg_short_b,
"conf/distance_table/short_b.bin",
"配置文件");

DEFINE_string(calibrator_cfg_short,
"conf/nodes/config/camera_short.yaml",
"配置文件");
DEFINE_string(calibrator_cfg_long,
"conf/nodes/config/camera_long.yaml",
"配置文件");

DEFINE_string(calibrator_cfg_distortTable,
"conf/nodes/config/distortTable.bin",
"配置文件");

DEFINE_string(lidar_map_parameter,
"conf/nodes/config/lidar_map_image.yaml",
"配置文件");

