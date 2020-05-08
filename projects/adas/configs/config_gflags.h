
#pragma once

#include "gflags/gflags.h"

// The directory which contains a group of related maps, such as base_map,
// sim_map, routing_topo_grapth, etc.

DECLARE_string(adas_cfg_interface_file);
DECLARE_string(adas_debug_output_dir);


DECLARE_int32(adas_camera_size);
DECLARE_int32(adas_lidar_size);



DECLARE_string(distance_cfg_long_a);
DECLARE_string(distance_cfg_long_b);
DECLARE_string(distance_cfg_short_a);
DECLARE_string(distance_cfg_short_b);


DECLARE_string(calibrator_cfg_short);
DECLARE_string(calibrator_cfg_long);
DECLARE_string(calibrator_cfg_distortTable);
DECLARE_string(lidar_map_parameter);
