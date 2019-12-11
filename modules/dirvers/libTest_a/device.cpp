
#include "watrix/aiwIoss.hh"

#pragma comment(lib,"aiwUtil.lib")


IOSS_STATUS AIW_CALL load(aiwDriverInfo_ptr driverObj)
{
	AIW_LOG_TRACE("Driver: %1%:%2% is loading......", driverObj->vendor.key.Data, driverObj->type.key.Data);

	return IOSS_SUCCESS;
 }

IOSS_STATUS AIW_CALL unload()
{
	AIW_LOG_TRACE("Driver:is unloading......");
	return IOSS_SUCCESS;
 }

IOSS_STATUS AIW_CALL start_device(aiwDeviceInfo_ptr handle)
{
	AIW_LOG_TRACE("Device: %1% is starting......", handle->k.Data);
	return IOSS_SUCCESS;
 }

IOSS_STATUS AIW_CALL stop_device(aiwDeviceInfo_ptr handle)
{
	AIW_LOG_TRACE("Device: %1% is stopping......", handle->k.Data);
	return IOSS_SUCCESS;
 }

IOSS_STATUS AIW_CALL write_device(aiwITEM_cptr tte, aiwVariant_cptr value)
{
	return IOSS_SUCCESS;
 }

 aiwVoid AIW_CALL update_tag(aiwITEM_cptr tte,aiwData_ptr pt,aiwTimeStamp_cptr now)
 {
	 return ;
 }

 IOSS_STATUS AIW_CALL address_translate(aiwITEM_ptr tte)
 {
	 return IOSS_SUCCESS;
 }

 IOSS_STATUS AIW_CALL dispatch(aiwDeviceInfo_ptr device,aiwInt32 majorCode,aiwInt32 param)
 {
	 return IOSS_SUCCESS;
 }
