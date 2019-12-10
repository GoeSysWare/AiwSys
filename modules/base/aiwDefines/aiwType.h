/*****************************************************************************
*  AiwSys	Basic tool library											 *
*  @copyright Copyright (C) 2016 aiwrge.Kuo  shuimujie_study@163.com.							 *
*                                                                            *
*  This file is part of aiwSys.												*
*                                                                            *
*  @file     aiwBaseType.h                                                   *
*  @brief    AiwSys					                                       *
*																			 *
*  
*  
*  @author   aiwrge.Kuo                                                      *
*  @email    shuimujie_study@163.com												 *
*  @version  1.0.0.1(版本号)                                                 *
*  @date     2016.8															 *
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
#ifndef  AIWSYS_BASE_BASICTYPE_H__
#define  AIWSYS_BASE_BASICTYPE_H__

#include <cstddef>
#include <string>



//基本数据类型
typedef std::nullptr_t		aiwNull;
typedef unsigned char		aiwBool;
typedef signed char			aiwInt8;
typedef unsigned char		aiwUInt8;
typedef signed short		aiwInt16;
typedef unsigned short		aiwUInt16;
typedef signed int			aiwInt32;
typedef unsigned int		aiwUInt32;
typedef signed long long	aiwInt64;
typedef unsigned long long	aiwUInt64;
typedef float				aiwFloat;
typedef double				aiwDouble;
typedef char				aiwChar;
typedef wchar_t				aiwWChar;
typedef unsigned char		aiwByte;
typedef void				aiwVoid;
typedef int					aiwAPIStatus;

typedef aiwUInt64			aiwTime;		//做系统运行时的实时处理时间
typedef  aiwTime* 			aiwTime_ptr;
typedef const aiwTime * 	aiwTime_cptr;

typedef aiwUInt64				aiwTimeStamp;	//做数据信息里的时间戳
typedef  aiwTimeStamp * 		aiwTimeStamp_ptr;
typedef const aiwTimeStamp * 	aiwTimeStamp_cptr;


#pragma pack(push)
#pragma pack(1)

typedef struct  __tag_Blob         //二进制信息数据类型
{
	aiwUInt32	Length;
	aiwByte*	Data;
} aiwBlob;
typedef struct  __tag_Img         //图片类型
{
	aiwUInt32	Length;
	aiwByte*	Data;
} aiwImg;
typedef struct  __tag_Vedio        //视频类型
{
	aiwUInt32	Length;
	aiwByte*	Data;
} aiwVedio;

typedef struct  __tag_Audio         //音频类型
{
	aiwUInt32	Length;
	aiwByte*	Data;
} aiwAudio ;
typedef struct  __tag_Radio         //雷达类型
{
	aiwUInt32	Length;
	aiwByte*	Data;
} aiwRadio;

//带长度的ASCII c格式的字符串，做存储、通信用
typedef struct  __tag_String
{
	aiwUInt32	Length;
	aiwChar*	Data;
} aiwString;
//带长度的ASCII c格式的字符串，做存储、通信用
typedef struct  __tag_AString
{
	aiwUInt32	Length;
	aiwChar*	Data;
} aiwAString;
//带长度的UNICODE c格式的字符串，做存储、通信用
typedef struct  __tag_WString
{
	aiwUInt32	Length;
	aiwWChar*	Data;
} aiwWString;


typedef	aiwChar*			aiwStr;			//C 格式ASCII c格式的字符串，做存储、通信用
typedef	aiwWChar*			aiwWStr;		//C 格式UNICODE格式的字符串，做存储、通信用
typedef std::string			aiwStdString;	//C++格式的ASCII字符串，做函数调用处理用
typedef std::wstring		aiwStdWString;	//C++格式的UNICODE字符串，做函数调用处理用

											//带长度的ASCII c格式的字符串，做存储、通信用
typedef struct  __tag_StringList
{
	aiwUInt32	Count;
	aiwStr*		List;
} aiwStringList;							//ASCII c格式的字符串列表，做存储、通信用

#ifdef _UNICODE
typedef aiwWStr 		aiwTStr;			//UNICODE字符串	 C
typedef wchar_t			aiwTChar;			//UNICODE字符
typedef aiwStdWString	aiwStdTString;		//UNICODE字符串  C++
#else
typedef aiwStr			aiwSTtr;			//ASCII字符串	 C
typedef char			aiwTChar;			//ASCII字符
typedef aiwStdString aiwStdTString;			//ASCII字符串 C++
#endif // __UNICODE

typedef struct  __tag_APIStatusList
{
	aiwUInt32		Count;
	aiwAPIStatus*	List;
} aiwAPIStatusList;


//C++11的强类型enum
typedef enum  class __tag_VARTYPE : aiwUInt8
{
	vTypeEmpty = 0,
	vTypeBool,
	vTypeInt8,
	vTypeUInt8,
	vTypeInt16,
	vTypeUInt16,
	vTypeInt32,
	vTypeUInt32,
	vTypeInt64,
	vTypeUInt64,
	vTypeFloat,
	vTypeDouble,
	vTypeChar,
	vTypeWChar,
	vTypeByte,
	vTypeString,
	vTypeAString,
	vTypeWString,
	vTypeTimeStamp,
	vTypeBlob,
    vTypeImg,
    vTypeVedio,
	vTypeAudio,
	vTypeRadio,
	vTypeMax
}aiwVarTypeEnum;


typedef struct  __tag_Variant
{
	aiwVarTypeEnum		varType;
	union
	{
		aiwBool			vBool;
		aiwInt8			vInt8;
		aiwUInt8		vUInt8;
		aiwInt16		vInt16;
		aiwUInt16		vUInt16;
		aiwInt32		vInt32;
		aiwUInt32		vUInt32;
		aiwInt64		vInt64;
		aiwUInt64		vUInt64;
		aiwFloat		vFloat;
		aiwDouble		vDouble;
		aiwChar			vChar;
		aiwWChar		vWChar;
		aiwByte			vByte;
		aiwString		vString;
		aiwAString		vAString;
		aiwWString		vWString;
		aiwTimeStamp	vTimeStamp;
		aiwBlob			vBlob;
		aiwImg			vImg;
		aiwVedio		vVedio;
		aiwAudio		vAudio;	
		aiwRadio		vRadio;
	};
}aiwVariant,*aiwVariant_ptr;
typedef const aiwVariant* aiwVariant_cptr;

typedef struct  __tag_VariantList
{
	aiwUInt32	ValueCount;
	aiwVariant*	ValueList;
}aiwVariantList,*aiwVariantList_ptr;


///  数据记录定义
typedef struct __tag_Data
{
	aiwTime				Time;				// 时间戳
	aiwVariant			Value;				// 值
	aiwUInt8			Quality;			// 质量戳
} aiwData,*aiwData_ptr;
typedef const aiwData* aiwData_cptr;

///	 数据记录列表
typedef struct __tag_DataList
{
	aiwUInt32			DataCount;			// 数据记录个数
	aiwData*			DataList;			// 数据记录列表
} aiwDataList,*aiwDataList_ptr;


#pragma pack(pop)
#endif