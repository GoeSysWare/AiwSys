

#include "projects/common/registerer/registerer.h"

namespace aiwsys {
namespace projects {
namespace common {

BaseClassMap &GlobalFactoryMap() {
  static BaseClassMap factory_map;
  return factory_map;
}

bool GetRegisteredClasses(
    const std::string &base_class_name,
    std::vector<std::string> *registered_derived_classes_names) {
  CHECK_NOTNULL(registered_derived_classes_names);
  BaseClassMap &map = GlobalFactoryMap();
  auto iter = map.find(base_class_name);
  if (iter == map.end()) {
    AERROR << "class not registered:" << base_class_name;
    return false;
  }
  for (auto pair : iter->second) {
    registered_derived_classes_names->push_back(pair.first);
  }
  return true;
}

}  
}  
}  