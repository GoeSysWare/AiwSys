#include <future>
#include <iostream>
#include <functional>     // std::ref
#include <thread>         // std::thread


bool is_prime(int x)
{
  for (int i=1; i<x; i++)
  {
    if (x % i == 0)
      return false;
  }
  return true;
}
void print_int (std::future<int>& fut) {
  int x = fut.get();
  std::cout << "value: " << x << '\n';
}

// count down taking a second for each value:
int countdown (int from, int to) {
  for (int i=from; i!=to; --i) {
    std::cout << i << '\n';
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  std::cout << "Lift off!\n";
  return from-to;
}


int main()
{
  std::future<bool> fut1 = std::async(is_prime, 700020007);
  std::cout << "please wait";
  std::chrono::milliseconds span(100);
  while (fut1.wait_for(span) != std::future_status::ready)
    std::cout << ".";
  std::cout << std::endl;

  bool ret1 = fut1.get();
  std::cout << "final result: " << ret1<< std::endl;


  std::promise<int> prom;                      // create promise

  std::future<int> fut = prom.get_future();    // engagement with future

  std::thread th1 (print_int, std::ref(fut));  // send future to new thread

  prom.set_value (10);                         // fulfill promise
                                               // (synchronizes with getting the future)
  th1.join();


  std::packaged_task<int(int,int)> tsk (countdown);   // set up packaged_task
  std::future<int> ret = tsk.get_future();            // get future

  std::thread th (std::move(tsk),10,0);   // spawn thread to count down from 10 to 0

  // ...

  int value = ret.get();                  // wait for the task to finish and get result

  std::cout << "The countdown lasted for " << value << " seconds.\n";

  th.join();

  return 0;

  return 0;
}