/*****************************************************************************
*  AiwSys	Basic tool library											     *
*  @copyright Copyright (C) 2019 aiwrge.Kuo  shuimujie_study@163.com.							 *
*                                                                            *
*  This file is part of AiwSys.												 *
*                                                                            *
*  @file     aiwErrors.h													 *
*  @brief    AiwSys的内部错误码定义                                          *
*																			 *
*  @author   aiwrge.Kuo                                                      *
*  @email    shuimujie_study@163.com												 *
*  @version  1.0.1.0(版本号)                                                 *
*  @date     2019.6															 *
*  @license																	 *
*                                                                            *
*----------------------------------------------------------------------------*
*  Remark         :															 *
*----------------------------------------------------------------------------*
*  Change History :                                                          *
*  <Date>     | <Version> | <Author>       | <Description>                   *
*----------------------------------------------------------------------------*
*  2019/06/24 | 1.0.1.1   | aiwrge.Kuo      | Create file                    *
*----------------------------------------------------------------------------*
*                                                                            *
*****************************************************************************/
#ifndef __AIWSYS_BASE_ERRORS_H__
#define __AIWSYS_BASE_ERRORS_H__


#define AIW_ERROR(modNum, errNum) (((unsigned short)(modNum)) << 16 | (unsigned short)(errNum))
#define AIW_NO_ERROR				0

#define AIW_SUCCESS(errCode) (((errCode)&0x8000)? __false : __true)
#define AIW_FAILED(errCode) (((errCode)&0x8000)? __true : __false)



typedef enum
{
	aiwERR_COMMON_MIN = -5000,					// 程序常见的错误起始码
	aiwERR_COMMON_NO_MEMORY,					//内存不足
	aiwERR_COMMON_DATATYPE,						//数据类型无效
	aiwERR_COMMON_DATACHANGE_FAILED,
	aiwERR_COMMON_PARAMETER_INVALID,			//参数无效
	aiwERR_COMMON_FILE_NOT_EXIST,				//文件不存在
	aiwERR_COMMON_FILE_FAILED,					//文件不完整
	aiwERR_COMMON_CHECK_FAILED,					//校验失败
	aiwERR_COMMON_MAX = -4000,					// 程序常见的错误最大码
} aiwErrorCode_COMMON;



typedef enum
{
	aiwERR_DB_MIN = -10000,						// 数据库通用起始错误码
	aiwERR_DB_VERSION_DIFFRENT,					// 数据库版本不一致

	aiwERR_DB_MAX = -5000,						// 数据库通用最大错误码
} aiwErrorCode_DB;



#endif