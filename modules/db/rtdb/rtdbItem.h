
/*****************************************************************************
*  AiwSys	Basic tool library											 *
*  @copyright Copyright (C) 2019 George.Kuo shuimujie_study@163.com.		*
*                                                                            *
*  This file is part of GeoSys.												*
*                                                                            *
*  @file     rtdbItem.h                                                   *
*  @brief						    *
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

#ifndef __AIWSYS_KERNEL_ITEM_H__
#define __AIWSYS_KERNEL_ITEM_H__
#include "watrix/aiwDefines.hh"
#include "watrix/aiwDdk.hh"


#define rtkm_description_length   64
#define rtkm_tag_address_length   256


typedef struct __tag_AIW_ITEM{

	aiwChar Description[rtkm_description_length];
	struct __tag_DEVICE_KEY	Device;
	aiwChar Address[rtkm_tag_address_length];
    struct __tag_DEVICE_INFO * DeviceObj; 
	aiwChar		BinaryAddress[64]; 
	aiwData 	BroadcastedValue;
	RTK_LIST_ENTRY	DeviceLink;	
	RTK_LIST_ENTRY	RefreshLink;

} aiwITEM, *aiwITEM_ptr;
typedef const aiwITEM * aiwITEM_cptr;


#endif