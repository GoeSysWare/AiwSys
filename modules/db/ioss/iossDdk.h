/*****************************************************************************
*  AiwSys	Basic tool library											 *
*  @copyright Copyright (C) 2019 George.Kuo shuimujie_study@163.com.		*
*                                                                            *
*  This file is part of GeoSys.												*
*                                                                            *
*  @file     aiwDdk.h                                                   *
*  @brief	Input Output subsystem的定义文件					    *
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

#ifndef __AIWSYS_IOSS_DDK_H__
#define __AIWSYS_IOSS_DDK_H__

#include "watrix/aiwDefines.hh"
#include "watrix/aiwUtil.hh"



#pragma pack(push)
#pragma pack(1)

#ifdef _MSC_VER
typedef struct __tag_DRIVER_INFO aiwDriverInfo, *aiwDriverInfo_ptr;
#else
struct __tag_DRIVER_INFO;
#endif

typedef struct __tag_AIW_ITEM  aiwITEM, *aiwITEM_ptr;
typedef const aiwITEM* aiwITEM_cptr;

typedef struct __tag_DEVICE_KEY{
	aiwChar Data[16];
} aiwDeviceKey, *aiwDeviceKey_ptr;
typedef const aiwDeviceKey *aiwDeviceKey_cptr;





typedef struct __tag_VENDOR_KEY{
	aiwChar Data[16];
}aiwVendorKey, *aiwVendorKey_ptr;
typedef const aiwVendorKey * aiwVendorKey_cptr;


typedef struct __tag_VENDOR_INFO{
	aiwVendorKey	key;
	aiwChar		description[128];
	aiwChar		reserved[16];
}aiwVendorInfo, *aiwVendorInfo_ptr;
typedef const aiwVendorInfo * aiwVendorInfo_cptr;



typedef struct __tag_DEVTYPE_KEY{
	aiwChar Data[16];
}aiwDeviceType, *aiwDeviceType_ptr;
typedef const aiwDeviceType * aiwDeviceType_cptr;



typedef struct __tag_DEVTYPE_INFO{
	aiwDeviceKey	key;
	aiwChar		description[64];
	aiwChar		reserved[16];
}aiwDeviceTypeInfo, *aiwDeviceTypeInfo_ptr;
typedef const aiwDeviceTypeInfo * aiwDeviceTypeInfo_cptr;



typedef struct __tag_DEVICE_INFO{
	aiwDeviceKey		k;
	struct __tag_DRIVER_INFO * d;//驱动信息指针
	aiwVendorInfo		v;
	aiwDeviceTypeInfo	t;
	aiwChar				parameter[256];
	aiwChar				address[64];

	aiwVoid				*OwnerField;

	aiwInt32			flags;
	RTK_LIST_ENTRY		tags;
	aiwInt32			error;
	aiwChar				reserved[44];
} aiwDeviceInfo, *aiwDeviceInfo_ptr;
typedef const aiwDeviceInfo * aiwDeviceInfo_cptr;
#pragma pack(pop)

typedef int IOSS_STATUS;

//这是兼容C++11std::function模式使用
typedef IOSS_STATUS (AIW_CALL IO_LOAD_PROC)(struct __tag_DRIVER_INFO * hdriver);
typedef IOSS_STATUS (AIW_CALL IO_UNLOAD_PROC)();
typedef IOSS_STATUS (AIW_CALL IO_START_DEVICE_PROC)(aiwDeviceInfo_ptr dev);
typedef IOSS_STATUS (AIW_CALL IO_STOP_DEVICE_PROC)(aiwDeviceInfo_ptr dev);
typedef IOSS_STATUS (AIW_CALL IO_ADDRESS_TRANSLATE_PROC)(aiwITEM_ptr tte);
typedef aiwVoid		(AIW_CALL IO_UPDATE_TAG_PROC)(aiwITEM_cptr tag, aiwData_ptr new_value,aiwTimeStamp_cptr now);
typedef IOSS_STATUS	(AIW_CALL IO_WRITE_DEVICE_PROC)(aiwITEM_cptr tte, aiwVariant_cptr value);
typedef IOSS_STATUS (AIW_CALL IO_DISPATCH_PROC)(aiwDeviceInfo_ptr device,aiwInt32 majorCode,aiwInt32 param);


//这是兼容C++函数指针模式使用
typedef IOSS_STATUS (AIW_CALL *IO_LOAD_CB)(struct __tag_DRIVER_INFO * hdriver);
typedef IOSS_STATUS (AIW_CALL *IO_UNLOAD_CB)();
typedef IOSS_STATUS (AIW_CALL *IO_START_DEVICE_CB)(aiwDeviceInfo_ptr dev);
typedef IOSS_STATUS (AIW_CALL *IO_STOP_DEVICE_CB)(aiwDeviceInfo_ptr dev);
typedef IOSS_STATUS (AIW_CALL *IO_ADDRESS_TRANSLATE_CB)(aiwITEM_ptr tte);
typedef aiwVoid		(AIW_CALL *IO_UPDATE_TAG_CB)(aiwITEM_cptr tag,aiwData_ptr new_value,aiwTimeStamp_cptr now);
typedef IOSS_STATUS (AIW_CALL *IO_WRITE_DEVICE_CB)(aiwITEM_cptr tte, aiwVariant_cptr value);
typedef IOSS_STATUS (AIW_CALL *IO_DISPATCH_CB)(aiwDeviceInfo_ptr device, aiwInt32 majorCode, aiwInt32 param);



#pragma pack(push)
#pragma pack(1)

typedef struct __tag_DRIVER_INFO {
	aiwVoid						*plugin_handle;
	IO_LOAD_CB				load;
	IO_UNLOAD_CB				unload;
	IO_START_DEVICE_CB		start_device;
	IO_STOP_DEVICE_CB			stop_device;
	IO_ADDRESS_TRANSLATE_CB	address_translate;
	IO_UPDATE_TAG_CB			update_tag;
	IO_WRITE_DEVICE_CB		write_device;
	IO_DISPATCH_CB			dispatch;
	aiwUInt32					device_count;
	aiwVendorInfo				vendor;
	aiwDeviceTypeInfo			type;
	aiwChar						description[128];
	aiwChar						parameter[256];
	aiwChar						dllname[256];
	aiwUInt64					version;

	aiwInt32					flags;
	aiwChar						reserved[56];
} aiwDriverInfo;
typedef const aiwDriverInfo * aiwDriverInfo_cptr;

#pragma pack(pop)

#ifdef _WIN32
#define IOSS_DLL_EXPORT __declspec(dllexport)
#else
#define IOSS_DLL_EXPORT
#endif


CDECL_BEGIN

IOSS_DLL_EXPORT IOSS_STATUS load(
	aiwDriverInfo_ptr driverObj
	);

IOSS_DLL_EXPORT IOSS_STATUS unload();

IOSS_DLL_EXPORT IOSS_STATUS start_device(
	aiwDeviceInfo_ptr handle
	);

IOSS_DLL_EXPORT IOSS_STATUS stop_device(
	aiwDeviceInfo_ptr handle
	);

IOSS_DLL_EXPORT IOSS_STATUS write_device(
	aiwITEM_cptr tte, 
	aiwVariant_cptr value
	);

IOSS_DLL_EXPORT aiwVoid update_tag(
	aiwITEM_cptr tte, 
	aiwData_ptr pt, 
	aiwTimeStamp_cptr now
	);

IOSS_DLL_EXPORT IOSS_STATUS address_translate(
	aiwITEM_ptr tte
	);

IOSS_DLL_EXPORT IOSS_STATUS AIW_CALL dispatch(
	aiwDeviceInfo_ptr device,
	aiwInt32 majorCode,
	aiwInt32 param
	);

CDECL_END

#endif