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

#include <memory>
#include <string>

#include "CivetServer.h"
#include "cyber/cyber.h"
#include "modules/common/status/status.h"
#include "projects/adas/component/webview/backend/handlers/image_handler.h"
#include "projects/adas/component/webview/backend/handlers/websocket_handler.h"

/**
 * @namespace apollo::dreamview
 * @brief apollo::dreamview
 */
namespace apollo {
namespace dreamview {

class Dreamview {
 public:
  ~Dreamview();

  apollo::common::Status Init();
  apollo::common::Status Start();
  void Stop();

 private:
  void TerminateProfilingMode();

  std::unique_ptr<cyber::Timer> exit_timer_;


  std::unique_ptr<CivetServer> server_;
  std::unique_ptr<WebSocketHandler> websocket_;
  std::unique_ptr<WebSocketHandler> map_ws_;
  std::unique_ptr<WebSocketHandler> point_cloud_ws_;
  std::unique_ptr<ImageHandler> image_;

};

}  // namespace dreamview
}  // namespace apollo
