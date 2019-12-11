/*****************************************************************************
*  GeoSysWare	Basic tool library											 *
*  @copyright Copyright (C) 2019 George.Kuo  guooujie@163.com.							 *
*                                                                            *
*  This file is part of GeoSys.  using C++ std::thread                       *
*                                                                            *
*  @file     uitlSetting.h                                                     *
*  @brief    配置文件处理类                                                      *
*																			 *
*  其中包含了事件封装
*
*  @author   George.Kuo                                                      *
*  @email    shuimujie_study@163.com												 *
*  @version  1.0.0.1(版本号)                                                 *
*  @date     2019.8															 *
*  @license																	 *
*                                                                            *
*----------------------------------------------------------------------------*
*  Remark         : 必须在支持C++11环境下编译通过                            *
*----------------------------------------------------------------------------*
*  Change History :                                                          *
*  <Date>     | <Version> | <Author>       | <Description>                   *
*----------------------------------------------------------------------------*
*  2019/08/24 | 1.0.1.1   | George.Kuo      | Create file                    *
*----------------------------------------------------------------------------*
*                                                                            *
*****************************************************************************/

#ifndef  AIWSYS_UTIL_SETTING_H__
#define  AIWSYS_UTIL_SETTING_H__

#include <string>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include "watrix/aiwDefines.hh"
using namespace std;
using namespace boost;
using namespace boost::property_tree;

using boost::property_tree::ptree;

template<typename Type>
Type  util_get_xml_setting(string key,Type defualtVal, string filename)
{
	ptree tree;
	xml_parser::read_xml(filename, tree,  xml_parser::no_comments | xml_parser::trim_whitespace);
	Type val = tree.get<Type>(key,defualtVal);
    return val;
}
 
template<typename Type>
aiwVoid  util_set_xml_setting(string key,Type Val, const  string filename)
{
	ptree tree;
	tree.put<Type>(key,Val);
	xml_parser::write_xml(filename, tree);

    return;
}
 
template<typename Type>
Type  util_get_ini_setting(string key,Type defualtVal, string filename)
{
	ptree tree;
	ini_parser::read_ini(filename, tree);
	Type val = tree.get<Type>(key,defualtVal);
    return val;
}


 
template<typename Type>
aiwVoid  util_set_ini_setting(string key,Type Val, string filename)
{
	ptree tree;
	tree.put<Type>(key,Val);
	ini_parser::write_ini(filename, tree);

    return ;
}
 
template<typename Type>
Type  util_get_json_setting(string key,Type defualtVal, string filename)
{
	ptree tree;
	json_parser::read_json(filename, tree);
	Type val = tree.get<Type>(key,defualtVal);
    return val;
}
 
template<typename Type>
aiwVoid  util_set_json_setting(string key,Type Val, string filename)
{
	ptree tree;
	tree.put<Type>(key,Val);
	json_parser::write_json(filename, tree);
    return ;
}

#endif
