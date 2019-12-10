/*****************************************************************************
*  AiwSys	Basic tool library											 *
*  @copyright Copyright (C) 2019 George.Kuo shuimujie_study@163.com.		*
*                                                                            *
*  This file is part of GeoSys.												*
*                                                                            *
*  @file     utilLog.h                                                   *
*  @brief	公共基础头文件					    *
*																			 *
*
*
*  @author   George.Kuo                                                      *
*  @email   shuimujie_study@163.com												 *
*  @version  1.0.1.0(版本号)                                                 *
*  @date     2019.6														 *
*  @license																	 *
*                                                                            *
*----------------------------------------------------------------------------*
*  Remark         : 必须在支持C++11环境下编译通过                            *
*----------------------------------------------------------------------------*
*  Change History :                                                          *
*  <Date>     | <Version> | <Author>       | <Description>                   *
*----------------------------------------------------------------------------*
*  2019/06/24 | 1.0.1.1   | aiwrge.Kuo      | Create file                    *
*----------------------------------------------------------------------------*
*                                                                            *
*****************************************************************************/

#ifndef __AIWSYS_UTIL_LOG_H__
#define __AIWSYS_UTIL_LOG_H__



#include <boost/log/common.hpp>
#include <boost/log/expressions.hpp>

#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>

#include <boost/log/attributes/timer.hpp>
#include <boost/log/attributes/named_scope.hpp>

#include <boost/log/sources/logger.hpp>

#include <boost/log/support/date_time.hpp>

#include <string>
using namespace std;

namespace logging = boost::log;
namespace sinks = boost::log::sinks;
namespace attrs = boost::log::attributes;
namespace src = boost::log::sources;
namespace expr = boost::log::expressions;
namespace keywords = boost::log::keywords;

// 这里定义了一个日志级别的enum，后面在日志输出时会是一个属性值
enum severity_level
{
	normal,
	notification,
	warning,
	trace,
	error,
	fatal
};




#define AIW_LOG(level,msg) \
do{			\
	util_output_log(level, msg);	\
} while (0);


#define AIW_LOG_NORMAL(...) 	AIW_LOG(severity_level::normal, util_string_format(__VA_ARGS__)) 
#define AIW_LOG_DEBUG(...) 		AIW_LOG(severity_level::notification, util_string_format(__VA_ARGS__)) 
#define AIW_LOG_WARNING(...) 	AIW_LOG(severity_level::warning, util_string_format(__VA_ARGS__)) 
#define AIW_LOG_TRACE(...) 		AIW_LOG(severity_level::trace, util_string_format(__VA_ARGS__)) 
#define AIW_LOG_ERROR(...) 		AIW_LOG(severity_level::error, util_string_format(__VA_ARGS__)) 
#define AIW_LOG_FATAL(...) 		AIW_LOG(severity_level::fatal, util_string_format(__VA_ARGS__)) 



CDECL_BEGIN
AIW_UTIL_API aiwVoid AIW_CALL util_output_log(severity_level le, string msg);

AIW_UTIL_API aiwInt32 AIW_CALL util_init_log();
AIW_UTIL_API aiwVoid AIW_CALL util_uninit_log();

CDECL_END

#endif
