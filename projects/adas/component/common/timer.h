#ifndef ADAS_COMMON_TIMER_H_
#define ADAS_COMMON_TIMER_H_

#include <sys/time.h>

namespace watrix
{
namespace projects
{
namespace adas
{

class Timer {
 public:
  Timer() { Tic(); }
  void Tic() { gettimeofday(&start_tv_, nullptr); }
  uint64_t Toc() {
    struct timeval end_tv;
    gettimeofday(&end_tv, nullptr);
    uint64_t elapsed = (end_tv.tv_sec - start_tv_.tv_sec) * 1000000 +
                       (end_tv.tv_usec - start_tv_.tv_usec);
    Tic();
    return elapsed;
  }

 protected:
  struct timeval start_tv_;
};

}  
}  
}  

#endif