/*****************************************************************************
*  AIWSys	Basic tool library											 *
*  @copyright Copyright (C) 2016 George.Kuo  guooujie@163.com.							 *
*                                                                            *
*  This file is part of GeoSys.  using C++ std::thread                       *
*                                                                            *
*  @file     utilVariant.h                                                    *
*  @brief    Variant处理类                                                   *
*																			 *
*
*
*  @author   George.Kuo                                                      *
*  @email    shuimujie_study@163.com											 *
*  @version  1.0.1.1(版本号)                                                 *
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
#ifndef  GEOSYS_UTILS_VARIANT_H__
#define  GEOSYS_UTILS_VARIANT_H__

#include "aiwType.hh"
#include "util.h"

CDECL_BEGIN


///	 为geoVariant变量申请空间并初始化,但不设置类型,默认为empty/0类型

GEO_UITLS_API geoAPIStatus  GEO_CALL NEW_Variant(geoVariant **ppVariant);

///		为geoVariant变量申请空间并初始化

 GEO_UITLS_API geoAPIStatus   GEO_CALL Variant_New(   PIN geoVarTypeEnum nDataType,
							  POUT geoVariant **ppVariant);


///		释放已申请的geoVariant变量空间
GEO_UITLS_API  geoAPIStatus GEO_CALL  Variant_Free(PIN POUT geoVariant **ppVariant);


///	清空geoVariant变量 初始化
GEO_UITLS_API  geoAPIStatus GEO_CALL Variant_Clear(PIN POUT geoVariant *pVariant);



///		复制geoVariant变量

GEO_UITLS_API  geoAPIStatus GEO_CALL Variant_Copy(   PIN geoVariant *pSource,
							   PIN POUT geoVariant *pDestination);


///		将ANSI字符串赋给geoVariant变量

GEO_UITLS_API  geoAPIStatus GEO_CALL Variant_AStrToVariant(	PIN geoStr strAStr,
										PIN POUT geoVariant *pVariant);


///		将Ugeoode字符串赋给geoVariant变量

GEO_UITLS_API  geoAPIStatus GEO_CALL Variant_UStrToVariant(	PIN geoWStr strUStr,
										PIN POUT geoVariant *pVariant);


///		将字符串赋给geoVariant变量

 GEO_UITLS_API geoAPIStatus GEO_CALL Variant_StrToVariant(    PIN geoStr strStr,
									   PIN POUT geoVariant *pVariant);


///		将二进制串赋给geoVariant变量

GEO_UITLS_API geoAPIStatus GEO_CALL Variant_BlobToVariant(	PIN geoByte *pByteList,
										PIN geoUInt32 nLength,
										PIN POUT geoVariant *pVariant);


 ///	比较两个geoVariant值是否相等

GEO_UITLS_API geoBool GEO_CALL Variant_Equal(geoVariant *pFirst, geoVariant *pSecond);


///		将数据从内存到Variant结构体，必须存入对应类型；
///     调用者负责保证类型统一，如果在一个类型为geoInt 的geoVariant 中存入一个
///     字符串或二进制块，结果是不可预知的！

GEO_UITLS_API geoUInt32 GEO_CALL mem_copyto_Variant( geoVariant *Pdest,
							void * Pmem,const unsigned length);

GEO_UITLS_API geoDouble GEO_CALL geoVariant_GetDouble (const geoVariant* value);

///两个Variant之间转化
GEO_UITLS_API geoAPIStatus GEO_CALL VariantTypeCast(geoVarTypeEnum DataType, geoVariant* var_src,geoVariant* var_dst );


CDECL_END

#endif
