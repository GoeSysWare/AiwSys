#include "precomp.h"

#include <boost/dll/import.hpp> 
#include <boost/dll/shared_library.hpp>
#include <boost/dll/library_info.hpp>
#include <boost/filesystem.hpp>
#include <boost/function.hpp>

using namespace boost::dll;

DRIVER_LIST g_Drivers;


aiwBool _load_module(aiwDriverInfo & driver)
{

	AIW_LOG_TRACE("Loading IO driver: %1%.", driver.dllname);
	boost::dll::fs::error_code ec;

    //1 加载前，先将驱动的标识设置为未加载
	driver.flags &= ~DRIVER_FLAG_LOADED;

	//2 加载设备驱动的dll
	boost::dll::fs::path lib_path;
	//lib_path = boost::filesystem::current_path();
	lib_path /= DEVICE_DIR;
	lib_path /= driver.dllname;

	//为了与driver的结构体定义相同，采用纯指针，不用智能指针之类的
	boost::dll::shared_library *dll_hanle  = new boost::dll::shared_library();
	dll_hanle->load(lib_path, ec);

	//如果失败，则尝试加.dll 或 .so
	if( !dll_hanle->is_loaded() ){
		lib_path += boost::dll::shared_library::suffix().string();
		dll_hanle->load(lib_path, ec);
	}
	//如果失败，则尝试加载当前工作目录下的dll
	if (!dll_hanle->is_loaded()) {
		boost::dll::fs::path lib_path_backup;
		//lib_path_backup = boost::filesystem::current_path();
		lib_path_backup /= driver.dllname;
		lib_path_backup += boost::dll::shared_library::suffix().string();
		dll_hanle->load(lib_path_backup, ec);
	}

	if(!dll_hanle->is_loaded()){
		AIW_LOG_ERROR("Cannot load driver: %1%.", driver.dllname);
		delete dll_hanle;
		return aiwFALSE;
	}
	
	driver.plugin_handle = (void*)dll_hanle;  //h为函数LoadLibrary()返回的设备驱动dll的句柄
	//4 驱动信息标识设为已加载
	driver.flags |= DRIVER_FLAG_LOADED;

	boost::function<IO_LOAD_PROC> aoe = dll_hanle->get<IO_LOAD_PROC>("load");//得到设备驱动dll导出函数load()的指针
	//aoe(&driver);
    
	//note:此处有点复杂，为了兼容类似C的驱动接口，必须这么做
	//george-kuo 2019.7



    //将dll导出函数指针赋给DRIVER_INFO   driver中
	//取得C++11的导出函数的function
	boost::function<IO_LOAD_PROC>  symlink_load = dll_hanle->get<IO_LOAD_PROC>("load");
	//将function转化为原始的函数指针,再存储起来
	driver.load = *symlink_load.target<IO_LOAD_CB>();

	boost::function<IO_UNLOAD_PROC>  symlink_unload = dll_hanle->get<IO_UNLOAD_PROC>("unload");
	driver.unload = *symlink_unload.target<IO_UNLOAD_CB>();

	boost::function<IO_START_DEVICE_PROC>  symlink_start_device = dll_hanle->get<IO_START_DEVICE_PROC>("start_device");
	driver.start_device = *symlink_start_device.target<IO_START_DEVICE_CB>();

	boost::function<IO_STOP_DEVICE_PROC>  symlink_stop_device = dll_hanle->get<IO_STOP_DEVICE_PROC>("stop_device");
	driver.stop_device = *symlink_stop_device.target<IO_STOP_DEVICE_CB>();

	boost::function<IO_ADDRESS_TRANSLATE_PROC>  symlink_translate = dll_hanle->get<IO_ADDRESS_TRANSLATE_PROC>("address_translate");
	driver.address_translate = *symlink_translate.target<IO_ADDRESS_TRANSLATE_CB>();

	boost::function<IO_UPDATE_TAG_PROC>  symlink_update_tag = dll_hanle->get<IO_UPDATE_TAG_PROC>("update_tag");
	driver.update_tag = *symlink_update_tag.target<IO_UPDATE_TAG_CB>();

	boost::function<IO_WRITE_DEVICE_PROC>  symlink_write_device = dll_hanle->get<IO_WRITE_DEVICE_PROC>("write_device");
	driver.write_device = *symlink_write_device.target<IO_WRITE_DEVICE_CB>();

	boost::function<IO_DISPATCH_PROC>  symlink_dispatch = dll_hanle->get<IO_DISPATCH_PROC>("dispatch");
	driver.dispatch = *symlink_dispatch.target<IO_DISPATCH_CB>();

	if(!driver.dispatch){
		boost::function<IO_DISPATCH_PROC>  symlink_dispatch = dll_hanle->get<IO_DISPATCH_PROC>("_dispatch@12");
		driver.dispatch = *symlink_dispatch.target<IO_DISPATCH_CB>();
	}



	AIW_LOG_TRACE("IO driver %s loaded at 0x%08x", driver.dllname, dll_hanle->native());

	return aiwTRUE;
}


aiwDriverInfo_ptr get_driver(aiwDeviceKey_cptr key)
{
	aiwDeviceInfo_ptr dev;
	dev = io_open_device(key);
	if(!dev){
		return 0;
	}
	return dev->d;
}

IOSS_API aiwDriverInfo_ptr AIW_CALL io_load_driver(
	aiwVendorKey_cptr v,
	aiwDeviceType_cptr t
	)
{		
	DRIVER_LIST::iterator it;
	aiwDriverInfo driver;
	aiwBool isFind = aiwFALSE;
	ptree pt_config;
	ptree pt_drv;

	//找到配置文件文件，取出驱动DLL名称
	boost::filesystem::path infofile = boost::filesystem::current_path();

	//清空驱动信息结构体
	memset(&driver, 0,sizeof(aiwDriverInfo));

	infofile /= DEVICE_DIR;
	infofile /= INFS_DIR;
	infofile /= string(v->Data)+ INFS_SUFFIX;


	//遍历 找到所有配置的driver的vendor
	xml_parser::read_xml(infofile.string(), pt_config);
	auto vendors = pt_config.get_child(IOSS_DRIVER_VENDORS_KEY);
	if (vendors.empty())
	{
		AIW_LOG_ERROR("Empty context in driver config:'s%'", infofile);
		return aiwNULL;
	}

	for (auto ven : vendors)
	{
		pt_drv = ven.second;
		if (pt_drv.size() < 2) continue;
		string name = pt_drv.get<string>(IOSS_DRIVER_NAME_KEY);
		//找到对应的类型
		if (name == string(t->Data))
		{
			isFind = aiwTRUE;
			break;
		}
	}
	//如果没有找到
	if (!isFind)
	{
		AIW_LOG_ERROR("Invalid driver:'%s'In driver config", v->Data);
		return aiwNULL;
	}
	
	string drv_module = pt_drv.get<string>(IOSS_DRIVER_MODULE_KEY);

	strncpy(driver.dllname, drv_module.data(), sizeof(driver.dllname)-1);
	driver.dllname[sizeof(driver.dllname) - 1] = 0;

	//若驱动信息已在驱动列表g_Drivers中存在，则返回该驱动信息
	for(auto it : g_Drivers){
		if( !strnicmp(it.dllname, driver.dllname, sizeof(it.dllname)) ){
			return &it;  
		}
	}	
	strncpy(driver.vendor.key.Data, v->Data, sizeof(aiwVendorKey));

	strncpy(driver.type.key.Data, t->Data, sizeof(aiwDeviceType));
    
	//驱动厂家信息
	string oem_desc_key = IOSS_DRIVER_VENDORS_KEY + string(".") + IOSS_DRIVER_OEM_KEY;
	string oem_desc = util_get_xml_setting<string>(oem_desc_key, "", infofile.string());
	strncpy(driver.vendor.description, oem_desc.data(), oem_desc.length());

	//驱动描述信息

	string dev_desc = pt_drv.get<string>(IOSS_DRIVER_DESC_KEY);

	strncpy(driver.type.description, dev_desc.data(), sizeof(aiwVendorKey)-1);
	driver.type.description[sizeof(aiwVendorKey) - 1] = 0;

	//加载驱动dll，将驱动信息填入driver，并调用驱动dll导出函数load()下载驱动
	if(_load_module(driver)){
		if(driver.load){
			driver.load(&driver);
		}
	}
	
	//将填充好的驱动信息driver插入驱动列表g_Drivers中
	it = g_Drivers.insert(g_Drivers.end(), driver);
	
	//插入失败则卸载驱动,一般不会发生
	if(it == g_Drivers.end()){
		io_unload_driver(&driver);
		return aiwNULL;
	}
	
	return &(*it);
}

/*
功能：卸载设备驱动，并删除其在驱动列表中的信息
*/
IOSS_API aiwBool AIW_CALL io_unload_driver(aiwDriverInfo_ptr driver)
{
	DRIVER_LIST::iterator it;
	AIW_LOG_TRACE("Unloading IO driver: %1%", driver->dllname);
	//1 在驱动列表中找到驱动信息
	for(it = g_Drivers.begin(); it != g_Drivers.end(); it++){
		if(&(*it) == driver){     
			break;
		}
	}
	//2 排除驱动列表中找不到驱动信息的情况
	if(it == g_Drivers.end()){
		AIW_LOG_DEBUG("unload a non-existing driver : 0x%08x", driver);
		return aiwFALSE;
	}
    
	//3 若系统中还有与该驱动对应的设备，则不能卸载驱动
	if(driver->device_count){
		AIW_LOG_ERROR("Driver %1% cannot unload with %2% devices active.", driver->dllname, driver->device_count);
		return aiwFALSE;
	}
    
	//4 调用驱动dll的导出函数unload()，卸载驱动
	if(driver->unload){
		if(driver->unload() == IOSS_FAILED){
			AIW_LOG_DEBUG("driver %s rejected unload request.", driver->dllname);
			return aiwFALSE;
		}
	}
    
    //5 卸载驱动dll
	if(driver->plugin_handle){
		boost::dll::shared_library *dll_hanle = (boost::dll::shared_library*)driver->plugin_handle;
		dll_hanle->unload();
		delete dll_hanle;
	}

	//6 释放g_Drivers中与driver相关的信息
	g_Drivers.erase(it);
	return aiwTRUE;
}

