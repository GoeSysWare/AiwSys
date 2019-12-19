
/*****************************************************************************
*  AiwSys	Basic tool library											 *
*  @copyright Copyright (C) 2019 George.Kuo shuimujie_study@163.com.		*
*                                                                            *
*  This file is part of GeoSys.												*
*                                                                            *
*  @file     util.h                                                   *
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
*  Remark         :                            *
*----------------------------------------------------------------------------*
*  Change History :                                                          *
*  <Date>     | <Version> | <Author>       | <Description>                   *
*----------------------------------------------------------------------------*
*  2019/06/24 | 1.0.1.1   | aiwrge.Kuo      | Create file                    *
*----------------------------------------------------------------------------*
*                                                                            *
*****************************************************************************/

#ifndef __AIWSYS_BASE_UTIL_H__
#define __AIWSYS_BASE_UTIL_H__


#ifdef _WIN32
	#if defined(LIBUTIL_EXPORTS)
		#define AIW_UTIL_API _declspec(dllexport)
	#else
		#define AIW_UTIL_API _declspec(dllimport)
	#endif
#else
	#define AIW_UTIL_API 
#endif

#ifndef SYSTEM_LOG_FILE 
#define SYSTEM_LOG_FILE "system.log"
#endif 



#endif
