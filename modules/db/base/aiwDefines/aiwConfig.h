/*****************************************************************************
*   AiwSys	Basic tool library											 *
*  @copyright Copyright (C) 2019 George.Kuo shuimujie_study@163.com.							 *
*                                                                            *
*  This file is part of GeoSys.												*
*                                                                            *
*  @file     aiwConfig.h                                                   *
*  @brief	AiwSys的一些系统业务相关的配置和设置都在此					    *
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

#ifndef  AIWSYS_BASE_CONFIG_H__
#define  AIWSYS_BASE_CONFIG_H__




#include "aiwVersion.h"
#include "aiwNameSpace.h"

#define rtkm_node_key_length   16

////如果编译器不支持C++11编译自动失败
//#if __cplusplus < 20131103L  
//#error "should use C++ 11 implementation"  
//#endif  

//GCC大于4.9 支持正则表达式
#if (__GNUC__ == 4 && __GNUC_MINOR__ < 9)
#error " Ensure your GCC compiler verion  4.9 and later "
#endif

#ifdef _WIN32
#pragma warning(disable:4996) //全部关掉
#pragma warning(disable:4819)
#endif // _WIN32



/**
 * @brief 系统相关路径和文件配置定义
 * 
 */
#define DEVICE_DIR      "devices"
#define CONFIG_DIR      "config"
#define VAR_DIR         "vars"
#define LOG_DIR         "logs"
#define INFS_DIR        "infs"

//文件名
#define CONFIG_FILENAME   "devices"

//文件尾缀
#define INFS_SUFFIX       ".xml"
#define LOG_SUFFIX        ".log"
#define CONFIG_SUFFIX     ".xml"


#endif