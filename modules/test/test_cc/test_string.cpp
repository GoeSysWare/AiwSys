#include <iostream>
#include <chrono>
#include <thread>
#include <string>
#include <memory>
#include <string.h>
int main()
{

    char *p = (char *)malloc(12);
    memcpy(p,"test",strlen("test"));
    std::string str(p,strlen("test"));

    std::cout<< "p addr "<< static_cast<const void *> (p) <<std::endl;
     std::cout<< "str addr "<<static_cast<const void *> (str.c_str() )<<std::endl;
     std::cout<< "str addr "<<static_cast<const void *> (str.data() )<<std::endl;
    return 0;
}