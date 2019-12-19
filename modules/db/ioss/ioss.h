
/*****************************************************************************
*  AiwSys	Basic tool library											 *
*  @copyright Copyright (C) 2019 George.Kuo shuimujie_study@163.com.		*
*                                                                            *
*  This file is part of GeoSys.												*
*                                                                            *
*  @file     ioss.h                                                   *
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

#ifndef __AIWSYS_KERNEL_IOSS_H__
#define __AIWSYS_KERNEL_IOSS_H__



#include "iossDdk.h"

#define IOSS_MODULE_NUM 'IO'

#define IOSS_ERROR(code) AIW_ERROR(IOSS_MODULE_NUM, (code))

#define IOSS_OBJECT_NOT_FOUND IOSS_ERROR(-1)

#define IOSS_DEVICE_NOT_STARTED IOSS_ERROR(-2)

#define IOSS_DRIVER_NOT_LOADED IOSS_ERROR(-3)

#define IOSS_SUCCESS	0
#define IOSS_FAILED		-1

#define DISP_POWER_STATE_CHANGED		0x00000001

#define DISP_DB_ADD_TAG					0x01000000
#define DISP_DB_DROP_TAG				0x01000001

#define DISP_RESERVED					0x00000030

#define DRIVER_FLAG_LOADED	(0x1<<0) 


#define DF_Active		(0x1 << 0)
#define DF_Deleted	(0x1 << 1)



#define IOSS_XML_DEVICES_KEY  "devices"
#define IOSS_XML_DEVICE_KEY  "device"
#define IOSS_XML_DEVICE_NAME_KEY  "name"
#define IOSS_XML_DEVICE_VENDOR_KEY  "vendor"
#define IOSS_XML_DEVICE_TYPE_KEY  "type"
#define IOSS_XML_DEVICE_DESC_KEY  "description"
#define IOSS_XML_DEVICE_ADDR_KEY  "address"
#define IOSS_XML_DEVICE_PARA_KEY  "parameters"

#define IOSS_DRIVER_VENDORS_KEY  "vendors"
#define IOSS_DRIVER_VENDOR_KEY  "vendor"
#define IOSS_DRIVER_NAME_KEY  "type"
#define IOSS_DRIVER_MODULE_KEY  "module"
#define IOSS_DRIVER_OEM_KEY  "oem"
#define IOSS_DRIVER_DESC_KEY  "description"
#define IOSS_DRIVER_VERSION_KEY  "version"

#ifdef _WIN32
	#if defined(LIBIOSS_EXPORTS)
		#define IOSS_API _declspec(dllexport)
	#else
		#define IOSS_API _declspec(dllimport)
	#endif
#else
	#define IOSS_API 
#endif

CDECL_BEGIN

IOSS_API aiwBool AIW_CALL init_ioss();

IOSS_API aiwVoid AIW_CALL uninit_ioss();

IOSS_API aiwUInt32 AIW_CALL ioss_get_devices(
	aiwDeviceInfo_ptr  buffer,
	aiwUInt32 *maxcount
	);

IOSS_API aiwDriverInfo_ptr AIW_CALL io_load_driver(
	aiwVendorKey_cptr v,
	aiwDeviceType_cptr t
	);



IOSS_API aiwBool AIW_CALL io_unload_driver(aiwDriverInfo_ptr drv);

IOSS_API aiwDeviceInfo_ptr AIW_CALL io_create_device(
	aiwDriverInfo_ptr driverObj,
	aiwDeviceKey_cptr key
	);

IOSS_API aiwBool AIW_CALL io_update_tag(
	aiwITEM_cptr tag,
	aiwData_ptr new_value,
	aiwTimeStamp_cptr now
	);

IOSS_API aiwBool AIW_CALL io_delete_device(aiwDeviceInfo_ptr device);

IOSS_API aiwBool AIW_CALL io_start_device(aiwDeviceInfo_ptr device);

IOSS_API aiwBool AIW_CALL io_stop_device(aiwDeviceInfo_ptr device);

IOSS_API aiwDeviceInfo_ptr AIW_CALL io_open_device(aiwDeviceKey_cptr key);

IOSS_API aiwBool AIW_CALL io_write_device(
	aiwStr name, 
	aiwVariant_cptr value
	);
IOSS_API aiwBool AIW_CALL io_write_device_ex(
	aiwStr name,
	aiwData_cptr value
	);

IOSS_API aiwDeviceInfo_ptr AIW_CALL io_probe_device(
	const aiwChar * dev_name,
	aiwBool bValidateTags
	);

CDECL_END


#endif