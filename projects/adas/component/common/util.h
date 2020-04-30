#ifndef ADAS_COMMON_UTIL_H_
#define ADAS_COMMON_UTIL_H_
#include "cyber/common/environment.h"
#include "cyber/common/file.h"
#include<string>
#include <iostream>
#include <iomanip>
namespace watrix
{
namespace projects
{
namespace adas
{
    std::string GetAdasWorkRoot();
    std::string GetFloatRound(double fValue, int bits);
    float GetFloatRound2(double fValue, int bits);

}
} // namespace projects
} // namespace watrix
#endif
