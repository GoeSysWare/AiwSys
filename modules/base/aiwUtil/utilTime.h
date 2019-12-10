/*****************************************************************************
*  GeoSysWare	Basic tool library											 *
*  @copyright Copyright (C) 2019 George.Kuo  guooujie@163.com.							 *
*                                                                            *
*  This file is part of GeoSys.  using C++ std::thread                       *
*                                                                            *
*  @file     uitlTime.h                                                     *
*  @brief    线程处理类                                                      *
*																			 *
*  其中包含了事件封装
*
*  @author   George.Kuo                                                      *
*  @email    shuimujie_study@163.com												 *
*  @version  1.0.0.1(版本号)                                                 *
*  @date     2019.8															 *
*  @license																	 *
*                                                                            *
*----------------------------------------------------------------------------*
*  Remark         : 必须在支持C++11环境下编译通过                            *
*----------------------------------------------------------------------------*
*  Change History :                                                          *
*  <Date>     | <Version> | <Author>       | <Description>                   *
*----------------------------------------------------------------------------*
*  2019/08/24 | 1.0.1.1   | George.Kuo      | Create file                    *
*----------------------------------------------------------------------------*
*                                                                            *
*****************************************************************************/

#ifndef  AIWSYS_UTIL_TIME_H__
#define  AIWSYS_UTIL_TIME_H__


#include <chrono>
#include "watrix/aiwDefines.hh"
#include "util.h"


using namespace std::chrono;



//时间戳零点
 extern  const  aiwTimeStamp  aiwTime_Zero;

 CDECL_BEGIN

//以毫秒为计量单位
//此计时以系统启动时间为基准，
//此时间可以得到系统启动后CPU的tick计数，表示系统启动了多少毫秒
//主要用在程序内部定时，调节系统时间对程序无影响的地方
 AIW_UTIL_API aiwTime   AIW_CALL util_get_steadytime();

//以毫秒为计量单位
//此时间可以得到当前系统时间
//主要用在时间戳上
 AIW_UTIL_API aiwTimeStamp  AIW_CALL util_get_systemtime();

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
	bool return_pointer_to_first_digit DEFAULT_PARAM(false));

 CDECL_END


 //毫秒的aiwTime转换为CPU的tick时间
//为标准函数和系统函数所用
 AIW_UTIL_API steady_clock::time_point   AIW_CALL util_aiwtime2steadytime(aiwTime absTime);
#endif