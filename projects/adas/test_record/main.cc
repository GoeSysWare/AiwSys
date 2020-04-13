/******************************************************************************
 * Copyright 2018 The Apollo Authors. All Rights Reserved.
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

#include <getopt.h>
#include <stddef.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "cyber/common/file.h"
#include "cyber/common/time_conversion.h"
#include "cyber/init.h"
#include "projects/adas/test_record/info.h"
#include "projects/adas/test_record/player/player.h"
#include "projects/adas/test_record/recorder.h"
#include "projects/adas/test_record/recoverer.h"
#include "projects/adas/test_record/spliter.h"

using apollo::cyber::common::GetFileName;
using apollo::cyber::common::StringToUnixSeconds;
using apollo::cyber::common::UnixSecondsToString;
using apollo::cyber::record::HeaderBuilder;
using apollo::cyber::record::Info;
using apollo::cyber::record::Player;
using apollo::cyber::record::PlayParam;
using apollo::cyber::record::Recorder;
using apollo::cyber::record::Recoverer;
using apollo::cyber::record::Spliter;



int main(int argc, char** argv) {
  std::string binary = GetFileName(std::string(argv[0]));


  std::vector<std::string> opt_file_vec;
  std::vector<std::string> opt_output_vec;
  std::vector<std::string> opt_white_channels;
  std::vector<std::string> opt_black_channels;
  bool opt_all = true;
  bool opt_loop = false;
  float opt_rate = 1.0f;
  uint64_t opt_begin = 0;
  uint64_t opt_end = UINT64_MAX;
  uint64_t opt_start = 0;
  uint64_t opt_delay = 0;
  uint32_t opt_preload = 3;
  auto opt_header = HeaderBuilder::GetHeader();

if (opt_white_channels.empty() && !opt_all) {
      std::cout
          << "MUST specify channels option (-c) or all channels option (-a)."
          << std::endl;
      return -1;
    }
    if (opt_output_vec.size() > 1) {
      std::cout << "TOO many output file option (-o)." << std::endl;
      return -1;
    }
    if (opt_output_vec.empty()) {
      std::string default_output_file =
          UnixSecondsToString(time(nullptr), "%Y%m%d%H%M%S") + ".record";
      opt_output_vec.push_back(default_output_file);
    }
    ::apollo::cyber::Init(argv[0]);
    auto recorder = std::make_shared<Recorder>(opt_output_vec[0], opt_all,
                                               opt_white_channels, opt_header);
    bool record_result = recorder->Start();
    if (record_result) {
      while (!::apollo::cyber::IsShutdown()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      record_result = recorder->Stop();
    }
    return record_result ? 0 : -1;
 

}
