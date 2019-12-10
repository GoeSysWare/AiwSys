#include "precomp.h"
#include <boost/filesystem.hpp>
#include <boost/function.hpp>


aiwBool pnp_init()
{
	return aiwTRUE;
}

aiwBool pnp_uninit()
{
	return aiwTRUE;
}


IOSS_API aiwDeviceInfo_ptr AIW_CALL io_probe_device(
	const aiwChar * dev_name,
	aiwBool bValidateTags
)
{	
	aiwDriverInfo_ptr pDriver = aiwNULL;
	aiwDeviceInfo_ptr pDevice = aiwNULL;

	aiwBool isFind = aiwFALSE;
	ptree pt_config;
	ptree pt_dev;
	//找到配置文件文件
	boost::filesystem::path infofile = boost::filesystem::current_path();

	infofile /= CONFIG_DIR;
	infofile /= CONFIG_FILENAME;
	infofile += CONFIG_SUFFIX;


	//找到所有配置的device
	xml_parser::read_xml(infofile.string(), pt_config);
	auto devices = pt_config.get_child(IOSS_XML_DEVICES_KEY);
	if (devices.empty())
	{
		AIW_LOG_ERROR("Here is empty context in configure file:'s%'", infofile);
		return aiwNULL;
	}


	for (auto d : devices)
	{
		pt_dev = d.second;
		string name = pt_dev.get<string>(IOSS_XML_DEVICE_NAME_KEY);
		//找到device
		if (name == string(dev_name))
		{
			isFind = aiwTRUE;
			break;
		}
	}
	//如果没有找到
	if (!isFind)
	{
		AIW_LOG_ERROR("Invalid device:'%s'In config", dev_name);
		return aiwNULL;
	}

	string vendor = pt_dev.get<string>(IOSS_XML_DEVICE_VENDOR_KEY);
	string type = pt_dev.get<string>(IOSS_XML_DEVICE_TYPE_KEY);
	string addr = pt_dev.get<string>(IOSS_XML_DEVICE_ADDR_KEY);
	string para = pt_dev.get<string>(IOSS_XML_DEVICE_PARA_KEY);

	aiwVendorKey vendor_key;
	aiwDeviceType  device_type;
	//安全字符串拷贝
	strncpy(vendor_key.Data, vendor.c_str(), sizeof(aiwVendorKey) - 1);
	strncpy(device_type.Data, type.c_str(), sizeof(aiwDeviceType) - 1);
	vendor_key.Data[sizeof(aiwVendorKey) - 1] = 0;
	device_type.Data[sizeof(aiwDeviceType) - 1] = 0;

	//在驱动配置文件中提取信息，赋给DRIVER_INF,再把DIRVER_INFO插入g_drivers，并返回
	pDriver = io_load_driver(&vendor_key, &device_type);

	if (IsNull(pDriver))
	{
		AIW_LOG_ERROR("Cannot load IO driver %s:%s", vendor, type);
	}
	//
	aiwDeviceKey device_key;
	strncpy(device_key.Data, dev_name, sizeof(aiwDeviceKey) - 1);
	device_key.Data[sizeof(aiwDeviceKey) - 1] = 0;

	//加载设备
	pDevice = io_create_device(pDriver,&device_key);

	if (IsNull(pDevice))
	{
		AIW_LOG_ERROR("Cannot create device %s", device_key.Data); 
		return aiwNULL;
	}

	//完善设备信息
	if (IsNotNull(pDriver))
	{
		pDevice->v = pDriver->vendor;
		pDevice->t = pDriver->type;
		strncpy(pDevice->address, addr.c_str(), sizeof(pDevice->address));
		pDevice->address[sizeof(pDevice->address) - 1] = 0;
		strncpy(pDevice->parameter, para.c_str(), sizeof(pDevice->parameter));
		pDevice->parameter[sizeof(pDevice->parameter) - 1] = 0;
	}

	//连接相关的标签
	if (bValidateTags) {
		attach_device_tags(pDevice);
	}

	//调用设备驱动力的start_device接口启动设备
	io_start_device(pDevice);
	
	return pDevice;
}

aiwBool pnp_probe_devices()
{


	//找到配置文件文件
	boost::filesystem::path infofile = boost::filesystem::current_path();

	infofile /= CONFIG_DIR;
	infofile /= CONFIG_FILENAME;
	infofile += CONFIG_SUFFIX;

	ptree pt;

	//找到所有配置的device
	xml_parser::read_xml(infofile.string(), pt);
	auto devices = pt.get_child(IOSS_XML_DEVICES_KEY);
	if (devices.empty())
		return aiwFALSE;

	for (auto &d : devices)
	{
		ptree dev = d.second;
		string name = dev.get<string>(IOSS_XML_DEVICE_NAME_KEY);
		//根据子属性添加设备
		io_probe_device(name.c_str(), aiwFALSE);
	}
	return aiwTRUE;
}

aiwBool pnp_stop_devices()
{
	aiwDeviceInfo_ptr dev = aiwNULL;

	DEVICE_LIST::iterator it;

	for (it = g_Devices.begin(); it != g_Devices.end(); ) {
		dev = it->second;
		//停止设备
		if (io_stop_device(dev)) {
			it++;
			io_delete_device(dev);//设备停止后，再将设备信息从系统设备列表中删除
		}
	}
	if (g_Drivers.size())
	{
		AIW_LOG_ERROR("%d drivers are still in memory after stopping command.", g_Drivers.size());
	}

	return aiwTRUE;
}