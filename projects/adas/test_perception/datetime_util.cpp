#include "datetime_util.h"

#include <boost/date_time/posix_time/posix_time.hpp>
using namespace std;
namespace watrix{
      namespace util{

        long DatetimeUtil::GetMillisec(){
            boost::posix_time::ptime start_time =boost::posix_time::microsec_clock::local_time();
            const boost::posix_time::time_duration td = start_time.time_of_day();
            long millisecond = td.total_milliseconds();// - ((td.hours() * 3600 + td.minutes() * 60 + td.seconds()) * 1000) + td.seconds()*1000;
            return millisecond;
        }
        
        string DatetimeUtil::GetDateTime(){
            string strPosixTime = boost::posix_time::to_iso_string(boost::posix_time::second_clock::local_time());
            string microsecTime =  boost::posix_time::to_iso_string(boost::posix_time::microsec_clock::universal_time());
            int dotpos = microsecTime.find('.');
            string ms = microsecTime.substr(dotpos,microsecTime.length());
            int pos = strPosixTime.find('T');
            strPosixTime.replace(pos,1,std::string("_"));
            strPosixTime.replace(pos + 3,0,std::string("_"));
            strPosixTime.replace(pos + 6,0,std::string("_"));
            strPosixTime.append(ms);
            //cout<<strPosixTime<<endl; //20190114-13:51:40.007244
	        return  strPosixTime; 
        }

        std::string DatetimeUtil::GetFloatRound(double fValue, int bits)
        {
            stringstream sStream;
            string out;
            sStream << fixed << setprecision(bits) << fValue;
            sStream >> out;
            return out;
        }

        float DatetimeUtil::GetFloatRound2(double fValue, int bits)
        {
            stringstream sStream;
            string out;
            sStream << fixed << setprecision(bits) << fValue;
            sStream >> out;
            float r = std::atof(out.c_str());
            return r;
        }
        void DatetimeUtil::SplitString(const std::string& s, std::vector<std::string>& v, const std::string& c)
        {
            std::string::size_type pos1, pos2;
            pos2 = s.find(c);
            pos1 = 0;
            while(std::string::npos != pos2)
            {
                v.push_back(s.substr(pos1, pos2-pos1));
                
                pos1 = pos2 + c.size();
                pos2 = s.find(c, pos1);
            }
            if(pos1 != s.length())
                v.push_back(s.substr(pos1));
        }

        void DatetimeUtil::GetFileTime(std::string name, long &second, long &file_full_time)
        {
            // file_full_time 秒毫秒
            //47129174.pcd 变为 47129174 毫秒
            
            ////////////////////////////////////////////////////
            ///////////////////////////////////////////////////
            std::string n1;
            std::string str_second;
            std::string::size_type position;
            // //find 函数 返回jk 在s 中的下标位置 
            position = name.find(".pcd");
            if (position != name.npos) //如果没找到，返回一个特别的标志c++中用npos表示，我这里npos取值是4294967295，
            {
                //1562834636.298073515_pc.pcd
                n1 = name.substr((name.length()-22), 5) + name.substr((name.length()-16), 3);
                str_second = name.substr((name.length()-22), 5);                                   
            }else{
                //ros 1562834636.040952516_imag.png
                n1 = name.substr((name.length()-24), 5) + name.substr((name.length()-18), 3);
                str_second = name.substr((name.length()-24), 5);                  
            }

            // if (position != name.npos) //如果没找到，返回一个特别的标志c++中用npos表示，我这里npos取值是4294967295，
            // {
            //     //1562834636.298073515_pc.pcd
            //     n1 = name.substr((name.length()-7), 3);
            //     str_second = name.substr((name.length()-12), 5);                                   
            // }else{
            //     //ros 1562834636.040952516_imag.png
            //     n1 = name.substr((name.length()-7), 3);
            //     str_second = name.substr((name.length()-12), 5);               
            // }


            second =  std::atol(str_second.c_str());
            // std::cout<<"second :"<<str_second<<std::endl;
            // std::cout<<"micro sec :"<<n1<<std::endl;
            // std::cout<<"------------------"<<std::endl;
            file_full_time = std::atol(n1.c_str());
            if(file_full_time<0){
                file_full_time=0;
            }		
        }


      }
}