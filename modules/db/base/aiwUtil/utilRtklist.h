
/*****************************************************************************
*   AiwSys	Basic tool library											 *
*  @copyright Copyright (C) 2019 George.Kuo shuimujie_study@163.com.							 *
*                                                                            *
*  This file is part of GeoSys.												*
*                                                                            *
*  @file     utilRtkList.h                                                   *
*  @brief					    					*
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
*  Remark         :                          *
*----------------------------------------------------------------------------*
*  Change History :                                                          *
*  <Date>     | <Version> | <Author>       | <Description>                   *
*----------------------------------------------------------------------------*
*  2019/06/24 | 1.0.1.1   | george.Kuo      | Create file                    *
*----------------------------------------------------------------------------*
*                                                                            *
*****************************************************************************/

#ifndef  AIWSYS_UTILS_RTKLIST_H__
#define  AIWSYS_UTILS_RTKLIST_H__



typedef struct _RTK_SINGLE_LIST_ENTRY {
	struct _RTK_SINGLE_LIST_ENTRY *Next;
} RTK_SINGLE_LIST_ENTRY, *PRTK_SINGLE_LIST_ENTRY;


typedef struct _RTK_LIST_ENTRY {
	struct _RTK_LIST_ENTRY *Flink;
	struct _RTK_LIST_ENTRY *Blink;
}RTK_LIST_ENTRY, *PRTK_LIST_ENTRY;

#ifdef RTK_FIELD_OFFSET
#undef RTK_FIELD_OFFSET
#endif


#define RTK_FIELD_OFFSET(type, field)	 ((int)(int*)&(((type *)0)->field))

#ifndef RTK_CONTAINING_RECORD
#define RTK_CONTAINING_RECORD(address, type, field) \
	((type *)( (char *)(address) - RTK_FIELD_OFFSET(type, field)))
#endif



#define SAFE_CONTAINING_RECORD(address, type, field) \
((address)? ((type *)( (char *)(address) - RTK_FIELD_OFFSET(type, field))) : 0)


#define RtkInitializeListHead(ListHead) (\
	(ListHead)->Flink = (ListHead)->Blink = (ListHead))


#define RtkMergeList(List1, List2) {\
	PRTK_LIST_ENTRY Tail_1;\
	PRTK_LIST_ENTRY Tail_2;\
	Tail_1 = (List1)->Blink;\
	Tail_2 = (List2)->Blink;\
	Tail_1->Flink = (List2);\
	(List2)->Blink = Tail_1;\
	Tail_2->Flink = (List1);\
	(List1)->Blink = Tail_2;\
	}


#define RtkIsListEmpty(ListHead) \
	((ListHead)->Flink == (ListHead))



#define RtkRemoveHeadList(ListHead) \
	(ListHead)->Flink;\
	{RtkRemoveEntryList((ListHead)->Flink)}



#define RtkRemoveTailList(ListHead) \
	(ListHead)->Blink;\
	{RtkRemoveEntryList((ListHead)->Blink)}



#define RtkRemoveEntryList(Entry) {\
	PRTK_LIST_ENTRY _EX_Blink;\
	PRTK_LIST_ENTRY _EX_Flink;\
	_EX_Flink = (Entry)->Flink;\
	_EX_Blink = (Entry)->Blink;\
	_EX_Blink->Flink = _EX_Flink;\
	_EX_Flink->Blink = _EX_Blink;\
	}



#define RtkInsertTailList(ListHead,Entry) {\
	PRTK_LIST_ENTRY _EX_Blink;\
	PRTK_LIST_ENTRY _EX_ListHead;\
	_EX_ListHead = (ListHead);\
	_EX_Blink = _EX_ListHead->Blink;\
	(Entry)->Flink = _EX_ListHead;\
	(Entry)->Blink = _EX_Blink;\
	_EX_Blink->Flink = (Entry);\
	_EX_ListHead->Blink = (Entry);\
	}


#define RtkInsertHeadList(ListHead,Entry) {\
	PRTK_LIST_ENTRY _EX_Flink;\
	PRTK_LIST_ENTRY _EX_ListHead;\
	_EX_ListHead = (ListHead);\
	_EX_Flink = _EX_ListHead->Flink;\
	(Entry)->Flink = _EX_Flink;\
	(Entry)->Blink = _EX_ListHead;\
	_EX_Flink->Blink = (Entry);\
	_EX_ListHead->Flink = (Entry);\
	}



#define RtkPopEntryList(ListHead) \
	(ListHead)->Next;\
	{\
		PRTK_SINGLE_LIST_ENTRY FirstEntry;\
		FirstEntry = (ListHead)->Next;\
		if (FirstEntry != NULL) {	  \
			(ListHead)->Next = FirstEntry->Next;\
		}							  \
	}



#define RtkPushEntryList(ListHead,Entry) \
	(Entry)->Next = (ListHead)->Next; \
	(ListHead)->Next = (Entry)



//	Triply linked list structure.  Can be used as either a list head, or
//	as link words.
//

typedef struct _TRIPLE_LIST_ENTRY {
	struct _TRIPLE_LIST_ENTRY *Flink;
	struct _TRIPLE_LIST_ENTRY *Blink;
	struct _TRIPLE_LIST_ENTRY *Head;
}TRIPLE_LIST_ENTRY, *PTRIPLE_LIST_ENTRY;


#define RtkInitializeTripleListHead(TripleListHead) (\
	(TripleListHead)->Flink = (TripleListHead)->Blink = (TripleListHead)->Head = (TripleListHead))

#define RtkIsTripleListEmpty(TripleListHead) \
	((TripleListHead)->Flink == (TripleListHead))



#define RtkRemoveHeadTripleList(TripleListHead) \
	(TripleListHead)->Flink;\
	{RtkRemoveEntryTripleList((TripleListHead)->Flink)}



#define RtkRemoveTailTripleList(TripleListHead) \
	(TripleListHead)->Blink;\
	{RtkRemoveEntryTripleList((TripleListHead)->Blink)}



#define RtkRemoveEntryTripleList(Entry) {\
	if( (Entry) != (Entry)->Head){\
		PTRIPLE_LIST_ENTRY _EX_Blink;\
		PTRIPLE_LIST_ENTRY _EX_Flink;\
		_EX_Flink = (Entry)->Flink;\
		_EX_Blink = (Entry)->Blink;\
		_EX_Blink->Flink = _EX_Flink;\
		_EX_Flink->Blink = _EX_Blink;\
		(Entry)->Head = (Entry);\
		(Entry)->Flink = (Entry);\
		(Entry)->Blink = (Entry);\
	}\
}



#define RtkInsertTailTripleList(TripleListHead,Entry) {\
	PTRIPLE_LIST_ENTRY _EX_Tail;\
	_EX_Tail = (TripleListHead)->Blink;\
	(Entry)->Flink = (TripleListHead);\
	(Entry)->Blink = _EX_Tail;\
	_EX_Tail->Flink = (Entry);\
	(TripleListHead)->Blink = (Entry);\
	(Entry)->Head = (TripleListHead);\
	}


#endif

