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


std::string  str_pad(unsigned int n)
{
			// 000001,000002
			std::stringstream ss;
			ss << std::setw(6) << std::setfill('0')<< n;
			return ss.str();
}

} // namespace adas
} // namespace projects
} // namespace watrix