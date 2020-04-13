

#pragma once

#include <memory>
#include <string>

#include "modules/perception/base/camera.h"
#include "modules/perception/camera/common/camera_frame.h"

#include "projects/common/registerer/registerer.h"
#include "base_init_options.h"

namespace aiwsys {
namespace projects {
namespace adas {



struct AdasDetectorInitOptions : public BaseInitOptions {
};

struct AdasDetectorOptions {};

class BaseAdasDetector {
 public:
  BaseAdasDetector() = default;

  virtual ~BaseAdasDetector() = default;

  virtual bool Init(const AdasDetectorInitOptions &options =
                        AdasDetectorInitOptions()) = 0;

  // @brief: detect obstacle from image.
  // @param [in]: options
  // @param [in/out]: frame
  // obstacle type and 2D bbox should be filled, required,
  // 3D information of obstacle can be filled, optional.
  virtual bool Detect(const AdasDetectorInitOptions &options,
                     apollo::perception::camera:: CameraFrame *frame) = 0;

  virtual std::string Name() const = 0;

  BaseAdasDetector(const BaseAdasDetector &) = delete;
  BaseAdasDetector &operator=(const BaseAdasDetector &) = delete;
};  // class BaseAdasDetector

PROJECTS_REGISTER_REGISTERER(BaseAdasDetector);

#define REGISTER_WEAR_DETECTOR(name) \
  PROJECTS_REGISTER_CLASS(BaseAdasDetector, name)

}

}
}