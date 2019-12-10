/*****************************************************************************
*  AiwSys	Basic tool library											     *
*  @copyright Copyright (C) 2019 George.Kuo  shuimujie_study@163.com.							 *
*                                                                            *
*  This file is part of AiwSys.												 *
*                                                                            *
*  @file     aiwMacro.h													     *
*  @brief                                             *
*																			 *
*  @author   George.Kuo                                                      *
*  @email    shuimujie_study@163.com												 *
*  @version  1.0.1.0(版本号)                                                 *
*  @date     2019.6														 *
*  @license																	 *
*                                                                            *
*----------------------------------------------------------------------------*
*  Remark         :															 *
*----------------------------------------------------------------------------*
*  Change History :                                                          *
*  <Date>     | <Version> | <Author>       | <Description>                   *
*----------------------------------------------------------------------------*
*  2019/06/24 | 1.0.1.1   | George.Kuo      | Create file                    *
*----------------------------------------------------------------------------*
*                                                                            *
*****************************************************************************/
#ifndef __AIWSYS_BASE_MACRO_H__
#define __AIWSYS_BASE_MACRO_H__

#include <boost/version.hpp>

/*
    定义工程使用boost时是否使用动态库链接的
    如果是静态链接则注释掉
 */
#define BOOST_ALL_DYN_LINK

/*
    定义了BOOST的自动链接LIB的功能，然后把需要的lib库在附加库里手动加入
    需要自动链接则注释掉
 */
// #define BOOST_ALL_NO_LIB 


#if BOOST_VERSION != 107000
#error " should use boost version  1.70"  
#endif  

#if defined(AIW_HEADER_INLINE)
#	define AIW_API inline
#else 
#	if defined(_MSC_VER) ||  defined(__BORLANDC__) || defined(__MINGW32__)
#		ifdef LIB_AIW_EXPORTS
#			define AIW_API      __declspec(dllexport)
#		else
#			define AIW_API      __declspec(dllimport)
#		endif
#	else
#define AIW_API
#	endif
#endif


//参数输入描述符
#define PIN
//参数输出描述符
#define POT

#ifdef _WIN32
#define AIW_CALL _stdcall
#else
#define AIW_CALL
#endif

//定义无用arg参数宏
#if !defined (AIW_UNUSED_ARG)
# if defined (__GNUC__) && ((__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 2))) || (defined (__BORLANDC__) && defined (__clang__))
#   define AIW_UNUSED_ARG(a) (void) (a)
# else 
#  define AIW_UNUSED_ARG(a) (a)
# endif 
#endif /* !AIW_UNUSED_ARG */



//设定标准C++的调用
#ifdef __cplusplus
#define CDECL_BEGIN	extern "C" {
#define CDECL_END	}
#define DEFAULT_PARAM(val)	= val
#else
#define CDECL_BEGIN
#define CDECL_END
#define DEFAULT_PARAM(val)
#endif







//AIWSys 浮点精度为8位有效数字 
//字面值常量默认为double型
#define AIW_EPSINON 0.00000001

#define AIW_OK   0
#define AIW_Fail -1

#define aiwTRUE  1
#define aiwFALSE 0

#define aiwNULL 0


#define IsNull(Ptr) (!Ptr)
#define IsNotNull(Ptr) (Ptr)

#if defined (_UNICODE) || defined (UNICODE)
#define AIW_TEXT(STR) L##STR
#else
#define AIW_TEXT(STR) STR
#endif


#endif