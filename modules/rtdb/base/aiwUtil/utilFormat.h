

/*****************************************************************************
*   AiwSys	Basic tool library											 *
*  @copyright Copyright (C) 2019 George.Kuo shuimujie_study@163.com.							 *
*                                                                            *
*  This file is part of GeoSys.												*
*                                                                            *
*  @file     utilFormat.h                                                   *
*  @brief					    					*
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
*  Remark         :                          *
*----------------------------------------------------------------------------*
*  Change History :                                                          *
*  <Date>     | <Version> | <Author>       | <Description>                   *
*----------------------------------------------------------------------------*
*  2019/06/24 | 1.0.1.1   | george.Kuo      | Create file                    *
*----------------------------------------------------------------------------*
*                                                                            *
*****************************************************************************/

#ifndef  AIWSYS_UTILS_FORMAT_H__
#define  AIWSYS_UTILS_FORMAT_H__

#include <boost/format.hpp>
#include <string>

inline string util_string_format(const char* str)
{
	return string(str);
}

template<class TFirst>
void util_string_format(boost::format& fmt, TFirst&& first)
{
	fmt % first;
}

template<class TFirst, class... TOther>
void util_string_format(boost::format& fmt, TFirst&& first, TOther&&... other)
{
	fmt % first;
	util_string_format(fmt, other...);
}

template<class TFirst, class... TOther>
std::string util_string_format(const char* format, TFirst&& first, TOther&&... other)
{
	boost::format fmt(format);
	util_string_format(fmt, first, other...);
	return fmt.str();
}
#endif