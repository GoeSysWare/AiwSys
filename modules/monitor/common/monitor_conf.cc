
#include "monitor_conf.h"
#include "cyber/common/file.h"
#include "cyber/common/log.h"
#include "modules/common/util/map_util.h"

#include "modules/common/util/string_util.h"
#include "modules/common/util/string_tokenizer.h"
#include "modules/monitor/proto/system_status.pb.h"


namespace apollo {
namespace monitor {


using apollo::common::util::StrAppend;
using apollo::common::util::StrCat;
using apollo::dreamview::HMIConfig;
using apollo::dreamview::HMIMode;
using google::protobuf::Map;

using  namespace apollo::dreamview;


DEFINE_string(hmi_modes_config_path, "/media/shuimujie/C14D581BDA18EBFA/10.Projects/01.Linux/02.github/AiwSys/modules/monitor/conf/",
              "HMI modes config path.");


constexpr char kNavigationModeName[] = "Navigation";
// Convert a string to be title-like. E.g.: "hello_world" -> "Hello World".
static std::string TitleCase(const std::string& origin) {
  static const std::string kDelimiter = "_";
  std::vector<std::string> parts =
      apollo::common::util::StringTokenizer::Split(origin, kDelimiter);
  for (auto& part : parts) {
    if (!part.empty()) {
      // Upper case the first char.
      part[0] = static_cast<char>(toupper(part[0]));
    }
  }
  return apollo::common::util::PrintIter(parts);
}
// List files by pattern and return a dict of {file_title: file_path}.
 static Map<std::string, std::string> ListFilesAsDict(const std::string& dir,
                                              const std::string& extension) {
  Map<std::string, std::string> result;
  const std::string pattern = StrCat(dir, "/*", extension);
  for (const std::string& file_path : cyber::common::Glob(pattern)) {
    // Remove the extension and convert to title case as the file title.
    const std::string filename = cyber::common::GetFileName(file_path);
    const std::string file_title =
        TitleCase(filename.substr(0, filename.length() - extension.length()));
    result.insert({file_title, file_path});
  }
  return result;
}


HMIConfig  LoadConfig() {
  HMIConfig config;
  // Get available modes, maps and vehicles by listing data directory.
  *config.mutable_modes() =
      ListFilesAsDict(FLAGS_hmi_modes_config_path, ".pb.txt");
  CHECK(!config.modes().empty())
      << "No modes config loaded from " << FLAGS_hmi_modes_config_path;

//   *config.mutable_maps() = ListDirAsDict(FLAGS_maps_data_path);
//   *config.mutable_vehicles() = ListDirAsDict(FLAGS_vehicles_config_path);
  AINFO << "Loaded HMI config: " << config.DebugString();
  return config;
}


HMIMode LoadMode(const std::string& mode_config_path) {
  HMIMode mode;
  CHECK(cyber::common::GetProtoFromFile(mode_config_path, &mode))
      << "Unable to parse HMIMode from file " << mode_config_path;
  // Translate cyber_modules to regular modules.
  for (const auto& iter : mode.cyber_modules()) {
    const std::string& module_name = iter.first;
    const CyberModule& cyber_module = iter.second;
    // Each cyber module should have at least one dag file.
    CHECK(!cyber_module.dag_files().empty())
        << "None dag file is provided for " << module_name << " module in "
        << mode_config_path;

    Module& module = LookupOrInsert(mode.mutable_modules(), module_name, {});
    module.set_required_for_safety(cyber_module.required_for_safety());

    // Construct start_command:
    //     nohup mainboard -p <process_group> -d <dag> ... &
    module.set_start_command("nohup mainboard");
    const auto& process_group = cyber_module.process_group();
    if (!process_group.empty()) {
      StrAppend(module.mutable_start_command(), " -p ", process_group);
    }
    for (const std::string& dag : cyber_module.dag_files()) {
      StrAppend(module.mutable_start_command(), " -d ", dag);
    }
    StrAppend(module.mutable_start_command(), " &");

    // Construct stop_command: pkill -f '<dag[0]>'
    const std::string& first_dag = cyber_module.dag_files(0);
    module.set_stop_command(StrCat("pkill -f \"", first_dag, "\""));
    // Construct process_monitor_config.
    module.mutable_process_monitor_config()->add_command_keywords("mainboard");
    module.mutable_process_monitor_config()->add_command_keywords(first_dag);
  }
  mode.clear_cyber_modules();
  AINFO << "Loaded HMI mode: " << mode.DebugString();
  return mode;
}




}
}
