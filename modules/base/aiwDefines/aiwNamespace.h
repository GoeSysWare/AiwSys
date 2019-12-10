/*****************************************************************************
*  AiwSys	Basic tool library											     *
*  @copyright Copyright (C) 2019 George.Kuo  shuimujie_study@163.com.							 *
*                                                                            *
*  This file is part of AiwSys.												 *
*                                                                            *
*  @file     aiwNamespace.h													 *
*  @brief							                                         *
*																			 *
*  @author   George.Kuo                                                      *
*  @email    shuimujie_study@163.com												 *
*  @version  1.0.1.0(版本号)                                                 *
*  @date     2019.8															 *
*  @license																	 *
*                                                                            *
*----------------------------------------------------------------------------*
*  Remark         :															 *
*----------------------------------------------------------------------------*
*  Change History :                                                          *
*  <Date>     | <Version> | <Author>       | <Description>                   *
*----------------------------------------------------------------------------*
*  2019/06/24 | 1.0.1.1   | George.Kuo      | Create file                    *
*----------------------------------------------------------------------------*
*                                                                            *
*****************************************************************************/
#ifndef __AIWSYS_BASE_NAMESPACE_H__
#define __AIWSYS_BASE_NAMESPACE_H__



#if defined (AIW_HAS_VERSIONED_NAMESPACE ) && AIW_HAS_VERSIONED_NAMESPACE == 1

# ifndef AIW_VERSIONED_NAMESPACE_NAME

#  define AIW_MAKE_VERSIONED_NAMESPACE_NAME_IMPL(MAJOR,MINOR,MICRO) AIW_ ## MAJOR ## _ ## MINOR ## _ ## MICRO
#  define AIW_MAKE_VERSIONED_NAMESPACE_NAME(MAJOR,MINOR,MICRO) AIW_MAKE_VERSIONED_NAMESPACE_NAME_IMPL(MAJOR,MINOR,MICRO)
#  define AIW_VERSIONED_NAMESPACE_NAME AIW_MAKE_VERSIONED_NAMESPACE_NAME(AIW_MAJOR_VERSION,AIW_MINOR_VERSION,AIW_MICRO_VERSION)
# endif  /* !AIW_VERSIONED_NAMESPACE_NAME */

# define AIW_BEGIN_VERSIONED_NAMESPACE_DECL NAMESPACE AIW_VERSIONED_NAMESPACE_NAME {
# define AIW_END_VERSIONED_NAMESPACE_DECL } \
  using NAMESPACE AIW_VERSIONED_NAMESPACE_NAME;

#else

# define AIW_VERSIONED_NAMESPACE_NAME
# define AIW_BEGIN_VERSIONED_NAMESPACE_DECL
# define AIW_END_VERSIONED_NAMESPACE_DECL

#endif  /* AIW_HAS_VERSIONED_NAMESPACE */

#endif  /* !AIW_VERSIONED_NAMESPACE_H */
