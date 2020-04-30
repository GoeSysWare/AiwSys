#include "projects/adas/component/common/util.h"

namespace watrix
{
namespace projects
{
namespace adas
{
std::string GetAdasWorkRoot()
{
    std::string work_root = apollo::cyber::common::GetEnv("ADAS_PATH");

    if (work_root.empty())
    {
        work_root = apollo::cyber::common::GetCurrentPath();
    }
    return work_root;
}

std::string GetFloatRound(double fValue, int bits)
{
    std::stringstream sStream;
    std::string out;
    sStream <<  std::fixed << std::setprecision(bits) << fValue;
    sStream >> out;
    return out;
}

float GetFloatRound2(double fValue, int bits)
{
     std::stringstream sStream;
     std::string out;
    sStream <<  std::fixed <<  std::setprecision(bits) << fValue;
    sStream >> out;
    float r = std::atof(out.c_str());
    return r;
}

} // namespace adas
} // namespace projects
} // namespace watrix