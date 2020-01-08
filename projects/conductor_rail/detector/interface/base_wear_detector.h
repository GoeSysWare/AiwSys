

#pragma once

#include <memory>
#include <string>

#include "modules/perception/base/camera.h"
#include "modules/perception/camera/common/camera_frame.h"
#include "modules/perception/camera/lib/interface/base_init_options.h"
#include "projects/common/registerer/registerer.h"


namespace aiwsys {
namespace projects {
namespace conductor_rail {



struct WearDetectorInitOptions : public apollo::perception::camera::BaseInitOptions {
  std::shared_ptr<apollo::perception::base::BaseCameraModel> base_camera_model = nullptr;
};

struct WearDetectorOptions {};

class BaseWearDetector {
 public:
  BaseWearDetector() = default;

  virtual ~BaseWearDetector() = default;

  virtual bool Init(const WearDetectorInitOptions &options =
                        WearDetectorInitOptions()) = 0;

  // @brief: detect obstacle from image.
  // @param [in]: options
  // @param [in/out]: frame
  // obstacle type and 2D bbox should be filled, required,
  // 3D information of obstacle can be filled, optional.
  virtual bool Detect(const WearDetectorInitOptions &options,
                     apollo::perception::camera:: CameraFrame *frame) = 0;

  virtual std::string Name() const = 0;

  BaseWearDetector(const BaseWearDetector &) = delete;
  BaseWearDetector &operator=(const BaseWearDetector &) = delete;
};  // class BaseObstacleDetector

PROJECTS_REGISTER_REGISTERER(BaseWearDetector);

#define REGISTER_WEAR_DETECTOR(name) \
  PROJECTS_REGISTER_CLASS(BaseObstacleDetector, name)

}

}
}