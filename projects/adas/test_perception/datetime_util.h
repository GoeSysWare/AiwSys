#pragma once

#include <vector>
#include <string>
#include <iostream>
namespace watrix
{
  namespace util
  {
	  class  DatetimeUtil{
    public:
      static long GetMillisec();
      static std::string GetDateTime();
      static std::string GetFloatRound(double fValue, int bits);
      static float GetFloatRound2(double fValue, int bits);
      static void SplitString(const std::string& s, std::vector<std::string>& v, const std::string& c);
      static void GetFileTime(std::string name, long &second, long &file_full_time);
    };
  }
}
