
#pragma once

#include "modules/view/proto/hmi_config.pb.h"
#include "modules/view/proto/hmi_mode.pb.h"

namespace apollo
{
namespace monitor
{

apollo::dreamview::HMIConfig LoadConfig();

apollo::dreamview::HMIMode LoadMode(const std::string &mode_config_path);

} // namespace monitor
} // namespace apollo
