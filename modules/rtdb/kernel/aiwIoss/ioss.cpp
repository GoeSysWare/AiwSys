
#include "precomp.h"
#include "ioss.h"
//static aiwBool AIW_API _power_callback(int newState, int context)
//{
//
//
//	return aiwTRUE;
//}

static void _load_settings()
{

}
IOSS_API aiwBool AIW_CALL init_ioss()
{


	_load_settings();

	
	pnp_probe_devices();

    // if(!CDBRefresher::init()){//add本地节点、组和标签；对g_Handlers[]中的函数指针赋值；开启一个线程
	// 	utils_error("Global initialization of CRefresher failed.\n");
	// 	return __false;
	// }	


	return aiwTRUE;
}

IOSS_API aiwVoid AIW_CALL uninit_ioss()
{



	pnp_stop_devices();


	//unregister_power_callback(_power_callback, 0);
	


}


static aiwBool _writeDevice(aiwITEM_ptr p, aiwVariant_cptr value)
{
	aiwDriverInfo_ptr npu;
	aiwBool retval = aiwFALSE;
	
	if(p->DeviceObj){
		npu = ((aiwDeviceInfo_ptr)p->DeviceObj)->d;
		if(npu){
			if(npu->write_device){
				retval = npu->write_device(p, value);
			}
		}
	}
	return retval;
}

IOSS_API aiwBool AIW_CALL io_write_device(
	aiwStr name,
	aiwVariant_cptr value
)
{
	aiwBool retval = aiwFALSE;
    
	////1 锁定
	//if(!lock_rtdb(__false, 100)){
	//	return __false;
	//}
	////2 找到本地节点中的指定标签点的RTK_TAG，其中存有该标签点对应的设备信息
	//PRTK_TAG p = query_tag(HNODE_LOCAL_MACHINE, name);
	//
	////3 将单点单实时值写入设备
	//if(p && !(p->d.Value.Flags & TF_ReadOnly)){
	//	//调用设备驱动导出函数，将单点单实时值写入设备
	//	retval = _writeDevice(p, value);
	//}
	////4 解锁
	//unlock_rtdb();

	return retval;
}
IOSS_API aiwBool AIW_CALL io_write_device_ex(
	aiwStr name,
	aiwData_cptr value
)
{
	return aiwTRUE;
}
IOSS_API aiwBool AIW_CALL io_update_tag(
	aiwITEM_cptr tag,
	aiwData_ptr new_value,
	aiwTimeStamp_cptr now
)
{
	//PDRIVER_INFO drv;
 //   
	////1 设置新实时值的Flags
	//new_value->Flags = 0;
	//set_value_type(new_value->Flags,  get_value_type(tag->s.Flags));
	//
	////2 将实时值的union清0，u64是union中的最大元素，即可表示union的大小
	///* 
	//pre-set all unused bytes to zero, this will make it easier to
	//write code depending on the tag types
	//*/
	//new_value->Value.u64 = 0;
	//
	//// assert(tag->d.DeviceObj == io_open_device(&tag->s.Device));

 //   //3 从标签的动态属性中得到设备驱动信息的结构体
	//if(!tag->d.DeviceObj){
	//	return __false;
	//}
	//drv = ((PDEVICE_INFO)tag->d.DeviceObj)->d;
	//if(!drv){
	//	return __false;
	//}

	////4 调用设备驱动DLL的导出函数更新实时值
	//if(!drv->update_tag){
	//	return __false;
	//}
	//
	//if(drv->update_tag){
	//	drv->update_tag(tag, new_value, now);
	//}
 //   
	////5 某些设备驱动可能会重写实时值的类型标识，需要再用标签静态属性中的类型标识设置一下
	//// some ill-behavioured driver will overwrite the type field
	//set_value_type(new_value->Flags, get_value_type(tag->s.Flags));
	//
	////6 若静态属性配置为开关量，则将true转化为1，false转化为0
	//if(get_value_type(tag->s.Flags) == dt_bool){
	//	/* make digital variable cannonical */
	//	new_value->Value.b = new_value->Value.b? 1 : 0;
	//}
	//
	return aiwTRUE;
}
