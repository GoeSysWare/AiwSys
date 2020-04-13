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


DEFINE_string(adas_camera_cfg,"./adas/production/conf/adas_camera.pb.txt","camera的配置文件");
DEFINE_int32(adas_camera_size,2,"ADAS支持的相机数");
DEFINE_bool(use_detect_model,false,"支持");
DEFINE_bool(use_train_seg_model,false,"支持");
DEFINE_bool(use_lane_seg_model,false,"支持");

DEFINE_string(distance_cfg_long_a,"cfg/autotrain_models/distance_table/long_a.bin","配置文件");
DEFINE_string(distance_cfg_long_b,"cfg/autotrain_models/distance_table/long_b.bin","配置文件");
DEFINE_string(distance_cfg_short_a,"cfg/autotrain_models/distance_table/short_a.bin","配置文件");
DEFINE_string(distance_cfg_short_b,"cfg/autotrain_models/distance_table/short_b.bin","配置文件");