
#ifndef __AIWSYS_IOSS_PRECOMP_H__
#define __AIWSYS_IOSS_PRECOMP_H__
#include "ioss.h"
#include "watrix/aiwDefines.hh"
#include "watrix/aiwUtil.hh"
#include "watrix/aiwRtdb.hh"


using namespace std;

aiwBool pnp_probe_devices();
aiwBool pnp_stop_devices();

typedef std::list<aiwDriverInfo> DRIVER_LIST;

typedef std::map<aiwDeviceKey, aiwDeviceInfo_ptr> DEVICE_LIST;

inline bool operator < (const aiwDeviceKey &lhs, const aiwDeviceKey &rhs) 
{
	return stricmp(lhs.Data, rhs.Data) < 0 ? true:false;
}

inline bool operator == (const aiwDeviceKey &lhs, const aiwDeviceKey &rhs)
{
	return stricmp(lhs.Data, rhs.Data) == 0 ? true : false;
}

extern DRIVER_LIST g_Drivers;
extern DEVICE_LIST g_Devices;



aiwDriverInfo_ptr get_driver(aiwDeviceKey_cptr key);

void attach_tag_to_device(aiwITEM_ptr tag, aiwDeviceInfo_ptr d);
void detach_tag_from_device(aiwITEM_cptr tag);
void attach_device_tags(aiwDeviceInfo_ptr  d);
void detach_device_tags(aiwDeviceInfo_ptr d);


#endif