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

#include "projects/adas/component/webview/backend/dreamview.h"

#include <vector>

#include "cyber/common/file.h"
#include "modules/common/time/time.h"
#include "projects/adas/component/webview/backend/common/dreamview_gflags.h"

namespace apollo {
namespace dreamview {

using apollo::common::Status;
using cyber::common::PathExists;

Dreamview::~Dreamview() { Stop(); }

void Dreamview::TerminateProfilingMode() {
  Stop();
  AWARN << "Profiling timer called shutdown!";
}

Status Dreamview::Init() {

  if (FLAGS_dreamview_profiling_mode &&
      FLAGS_dreamview_profiling_duration > 0.0) {
    exit_timer_.reset(
        new cyber::Timer(FLAGS_dreamview_profiling_duration,
                         [this]() { this->TerminateProfilingMode(); }, false));

    exit_timer_->Start();
    AWARN << "============================================================";
    AWARN << "| Dreamview running in profiling mode, exit in "
          << FLAGS_dreamview_profiling_duration << " seconds |";
    AWARN << "============================================================";
  }

  // Initialize and run the web server which serves the dreamview htmls and
  // javascripts and handles websocket requests.
  std::vector<std::string> options = {
      "document_root",      FLAGS_static_file_dir,   "listening_ports",
      FLAGS_server_ports,   "websocket_timeout_ms",  FLAGS_websocket_timeout_ms,
      "request_timeout_ms", FLAGS_request_timeout_ms};
  if (PathExists(FLAGS_ssl_certificate)) {
    options.push_back("ssl_certificate");
    options.push_back(FLAGS_ssl_certificate);
  } else if (FLAGS_ssl_certificate.size() > 0) {
    AERROR << "Certificate file " << FLAGS_ssl_certificate
           << " does not exist!";
  }
  server_.reset(new CivetServer(options));



  image_.reset(new ImageHandler());
  server_->addHandler("/image", *image_);

  return Status::OK();
}

Status Dreamview::Start() {


  return Status::OK();
}

void Dreamview::Stop() {
  server_->close();

}

}  // namespace dreamview
}  // namespace apollo
