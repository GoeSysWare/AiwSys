#include "watrix/aiwDefines.hh"
#include "util.h"
#include "utilLog.h"

#include <iostream>



using boost::shared_ptr;

static src::severity_logger< severity_level > slg;




// 定义上面的级别输出流操作的重载函数，在日志输出时会用到
template< typename CharT, typename TraitsT >
inline std::basic_ostream< CharT, TraitsT >& operator<< (
	std::basic_ostream< CharT, TraitsT >& strm, severity_level lvl)
{
	static const char* const str[] =
	{  // 这里的每一个值要与severity_level enum对应
		"normal",
		"notification",
		"warning",
		"error",
		"critical"
	};
	//如果日志的级别在enum里，则输出相应的文本
	if (static_cast<std::size_t>(lvl) < (sizeof(str) / sizeof(*str)))
		strm << str[lvl];
	else  //否则直接输出数字值
		strm << static_cast<int>(lvl);
	return strm;
}

AIW_UTIL_API aiwVoid AIW_CALL util_output_log(severity_level le, string msg)
{
	try
	{
		BOOST_LOG_SEV(slg, le) << msg;
	}
    catch (std::exception& e)
    {
        std::cout << "FAILURE: " << e.what() << std::endl;
        return;
    }

}

AIW_UTIL_API aiwInt32 AIW_CALL util_init_log()
{
	logging::add_console_log(std::clog, keywords::format = "%TimeStamp%: %Message%");

	logging::add_file_log
	(
		SYSTEM_LOG_FILE,
		keywords::filter = expr::attr< severity_level >("Severity") >= warning,
		keywords::format = expr::stream
		<< expr::format_date_time< boost::posix_time::ptime >("TimeStamp", "%Y-%m-%d, %H:%M:%S.%f")
		<< " [" << expr::format_date_time< attrs::timer::value_type >("Uptime", "%O:%M:%S")
		<< "] [" << expr::format_named_scope("Scope", keywords::format = "%n (%f:%l)")
		<< "] <" << expr::attr< severity_level >("Severity")
		<< "> " << expr::message
		//keywords::format = expr::format("%1% [%2%] [%3%] <%4%> %5%")
		//	% expr::format_date_time< boost::posix_time::ptime >("TimeStamp", "%Y-%m-%d, %H:%M:%S.%f")
		//	% expr::format_date_time< attrs::timer::value_type >("Uptime", "%O:%M:%S")
		//	% expr::format_named_scope("Scope", keywords::format = "%n (%f:%l)")
		//	% expr::attr< severity_level >("Severity")
		//	% expr::message
		
	);
	BOOST_LOG_FUNCTION();
	logging::add_common_attributes();
	logging::core::get()->add_thread_attribute("Scope", attrs::named_scope());
	slg.add_attribute("Uptime", attrs::timer());
	return 0;
}


AIW_UTIL_API aiwVoid AIW_CALL util_uninit_log()
{
    
}