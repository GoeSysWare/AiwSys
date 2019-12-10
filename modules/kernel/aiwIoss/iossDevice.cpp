

#include "precomp.h"
DEVICE_LIST	g_Devices;




void attach_tag_to_device(aiwITEM_ptr tag, aiwDeviceInfo_ptr d)
{

}
void detach_tag_from_device(aiwITEM_cptr tag)
{

}
void attach_device_tags(aiwDeviceInfo_ptr  d)
{

}
void detach_device_tags(aiwDeviceInfo_ptr d)
{

}



IOSS_API aiwDeviceInfo_ptr AIW_CALL io_create_device(
	aiwDriverInfo_ptr driverObj,
	aiwDeviceKey_cptr key
)
{
	aiwDeviceInfo_ptr dev = aiwNULL;
	dev = (aiwDeviceInfo_ptr)new char[sizeof(aiwDeviceInfo)];
	if (IsNull(dev)) {
		return aiwNULL;
	}
	std::memset(dev, 0, sizeof(aiwDeviceInfo));

	//填入设备名称
	dev->k = *key;

	// 填入设备驱动信息
	dev->d = driverObj;

	if (driverObj) {
		driverObj->device_count++;
	}
	//初始化设备信息的链接
	RtkInitializeListHead(&dev->tags);

	//插入设备列表
	g_Devices.insert(DEVICE_LIST::value_type(*key, dev));

	return dev;
}

IOSS_API aiwBool AIW_CALL io_delete_device(aiwDeviceInfo_ptr dev)
{
	aiwDriverInfo_ptr driverObj;
	DEVICE_LIST::iterator it;

	//解除设备与相关标签点的连接
	detach_device_tags(dev);

	//释放g_Devices中该设备信息
	it = g_Devices.find(dev->k);
	//如果没有此设备，也返回删除成功
	if (it == g_Devices.end())
	{
		return aiwTRUE;
	}
	//释放前需找到该设备信息关联的驱动信息的地址
	driverObj = dev->d;

	g_Devices.erase(it);
	delete dev;

	//驱动信息的处理
	if (driverObj) {
		//驱动信息中的关联设备数量减1
		driverObj->device_count--;
		assert(driverObj->device_count >= 0);
		//若发现驱动信息中的关联设备数量为0，则写在驱动DLL
		if (!driverObj->device_count) {
			io_unload_driver(driverObj);
		}
	}
	return aiwTRUE;
}


IOSS_API aiwUInt32 AIW_CALL ioss_get_devices(
	aiwDeviceInfo_ptr  buffer,
	aiwUInt32 *maxcount
)
{
	aiwUInt32 count, i;
	count = std::min((aiwUInt32)g_Devices.size(), *maxcount);
	DEVICE_LIST::iterator p;
	p = g_Devices.begin();
	for (i = 0; i < count; i++, p++) {
		buffer[i] = *(p->second);
	}
	if (*maxcount < g_Devices.size()) {
		*maxcount = g_Devices.size();
	}
	return count;
}

IOSS_API aiwDeviceInfo_ptr AIW_CALL io_open_device(aiwDeviceKey_cptr key)
{
	DEVICE_LIST::iterator it;
	it = g_Devices.find(*key);
	if (it == g_Devices.end()) {
		return aiwNULL;
	}
	return it->second;
}

IOSS_API aiwBool AIW_CALL io_start_device(aiwDeviceInfo_ptr dev)
{

	aiwBool r = aiwFALSE;
	IOSS_STATUS st = IOSS_FAILED;

	assert(dev->d);

	if (!(dev->d->flags & DRIVER_FLAG_LOADED)) {
		dev->error = IOSS_DRIVER_NOT_LOADED;
		r = aiwFALSE;
	}
	else {
		if (IsNull(dev->d->start_device)) {
			r = aiwTRUE;
		}
		else {
			st = dev->d->start_device(dev);
			if (st == IOSS_FAILED) {
				r = aiwFALSE;
				dev->error = IOSS_DEVICE_NOT_STARTED;
			}
			else
			{
				r = aiwTRUE;
			}
		}
	}
	//启动失败
	if (!r) {
		dev->flags &= ~DF_Active;
	}
	//启动成功
	else {
		dev->flags |= DF_Active;
		dev->error = 0;
	}

	//输出日志
	if (r) {
		AIW_LOG_TRACE("Device %s started.",dev->k.Data);
	}
	else {
		AIW_LOG_TRACE("Cannot start device %s.", dev->k.Data);
	}

	return r;	
}

IOSS_API aiwBool AIW_CALL io_stop_device(aiwDeviceInfo_ptr dev)
{
	if (!(dev->flags & DF_Active)) {
		//设备已经停止
		return aiwTRUE;
	}
	AIW_LOG_TRACE("Stopping device %s.", dev->k.Data);

	if (IsNotNull(dev->d)) {
		if (IsNotNull(dev->d->stop_device)) {

			if (dev->d->stop_device(dev)== IOSS_FAILED) {
				dev->flags &= ~DF_Active;
				AIW_LOG_TRACE("Device %s not responding to STOP command.", dev->k.Data);
				return aiwFALSE;
			}
		}
	}
	dev->flags &= ~DF_Active;
	AIW_LOG_TRACE("Device %s stopped.", dev->k.Data);
	return aiwTRUE;
}

