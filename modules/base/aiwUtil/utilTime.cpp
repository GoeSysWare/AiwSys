

#include"utilTime.h"

#include "boost/date_time/local_time/local_time.hpp"

const aiwTimeStamp  aiwTime_Zero = 0;



//以毫秒为计量单位
//此计时以系统启动时间为基准，
//此时间可以得到系统启动后CPU的tick计数，表示系统启动了多少毫秒
//主要用在程序内部定时，调节系统时间对程序无影响的地方
AIW_UTIL_API aiwTime   AIW_CALL util_get_steadytime()
{
	return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();

}
//以毫秒为计量单位
//此时间可以得到当前系统时间
//主要用在时间戳上
AIW_UTIL_API aiwTimeStamp  AIW_CALL util_get_systemtime()
{
	return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

}

//毫秒的aiwTime转换为CPU的tick时间
//为标准函数和系统函数所用
AIW_UTIL_API steady_clock::time_point   AIW_CALL util_aiwtime2steadytime(aiwTime absTime)
{

	return steady_clock::time_point(steady_clock::duration(milliseconds(absTime)));
}



/*
* 将aiwTimeStamp 格式化输出
* 格式为时:分:秒:毫秒的时间戳，ISO-8601标准格式
* 012345678901234567890123456
* 2010-12-02 12:56:00.123456<nul>
* 不直接用time_t是因为没有毫秒
* 格式化输出缓冲区最小24字节
*/
AIW_UTIL_API aiwTChar * AIW_CALL util_timestamp(const aiwTimeStamp& time_value,
	aiwTChar date_and_time[],
	size_t date_and_timelen,
	bool return_pointer_to_first_digit)
{
	if (date_and_timelen < 24)
	{
		errno = EINVAL;
		return 0;
	}

	aiwTimeStamp cur_time =
		(time_value == aiwTime_Zero) ?
		util_get_systemtime() : time_value;
	std::time_t secs = cur_time / 1000;

	struct std::tm *tms= std::localtime(&secs);

	snprintf(date_and_time,
		date_and_timelen,
		AIW_TEXT("%4.4d-%2.2d-%2.2d %2.2d:%2.2d:%2.2d.%03ld"),
		tms->tm_year + 1900,
		tms->tm_mon + 1,
		tms->tm_mday,
		tms->tm_hour,
		tms->tm_min,
		tms->tm_sec,
		static_cast<long> (cur_time % 1000));
	date_and_time[date_and_timelen - 1] = '\0';
	return &date_and_time[10 + (return_pointer_to_first_digit != 0)];
}