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

#pragma once

#include "gflags/gflags.h"

// The directory which contains a group of related maps, such as base_map,
// sim_map, routing_topo_grapth, etc.

DECLARE_string(adas_camera_cfg);
DECLARE_int32(adas_camera_size);
DECLARE_bool(use_detect_model);
DECLARE_bool(use_train_seg_model);
DECLARE_bool(use_lane_seg_model);


DECLARE_string(distance_cfg_long_a);
DECLARE_string(distance_cfg_long_b);
DECLARE_string(distance_cfg_short_a);
DECLARE_string(distance_cfg_short_b);

