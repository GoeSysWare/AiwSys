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
#include "modules/monitor/monitor.h"

#include "modules/common/time/time.h"
#include "modules/monitor/common/monitor_manager.h"
#include "modules/monitor/hardware/resource_monitor.h"
// #include "modules/monitor/software/channel_monitor.h"
#include "modules/monitor/software/process_monitor.h"
// #include "modules/monitor/software/recorder_monitor.h"
#include "modules/monitor/software/summary_monitor.h"

namespace apollo {
namespace monitor {

bool Monitor::Init() {
  MonitorManager::Instance()->Init(node_);




  // Monitor if processes are running.
  runners_.emplace_back(new ProcessMonitor());
  // Monitor if channel messages are updated in time.
  // runners_.emplace_back(new ChannelMonitor());
  // Monitor if resources are sufficient.
  runners_.emplace_back(new ResourceMonitor());
  // Monitor all changes made by each sub-monitor, and summarize to a final
  // overall status.
  runners_.emplace_back(new SummaryMonitor());

  return true;
}

bool Monitor::Proc() {
  const double current_time = apollo::common::time::Clock::NowInSeconds();
  if (!MonitorManager::Instance()->StartFrame(current_time)) {
    return false;
  }
  for (auto& runner : runners_) {
    runner->Tick(current_time);
  }
  MonitorManager::Instance()->EndFrame();

  return true;
}

}  // namespace monitor
}  // namespace apollo
