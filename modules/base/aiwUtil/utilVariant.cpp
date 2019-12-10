
#include <assert.h>
#include "preCompile.h"
#include "geoVariant.h"
#include "geoError.h"
#include "geoBasicType.h"


#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif
//
#define  DOUBLE_EXCEPTION			0.0001
#define  PSVARIANT_SIZE				sizeof(geoVariant)
#define  UINT32_SIZE				sizeof(geoUInt32)
#define  DOUBLE_SIZE				sizeof(geoDouble)
#define  FLOAT_SIZE					sizeof(geoFloat)
typedef geoBool (*PSVAEIANT_EQUAL)(geoVariant*,geoVariant*);
typedef geoAPIStatus (*VARIANT_CHANGETYPE)(geoVariant*);
#pragma warning(disable:4244 4996)



geoAPIStatus GEO_CALL NewMemory(void** ppDest, geoUInt32 nSize)
{

	*ppDest = geoNULL;
	*ppDest = malloc(nSize);
	if (*ppDest==geoNULL)
	{
		return geoERR_COMMON_NO_MEMORY;
	}
	return geoRET_SUCC;
}

 geoAPIStatus GEO_CALL NewAndClear(void** ppDest, geoUInt32 nSize)
{
	if (geoFail(NewMemory(ppDest,nSize)))
	{
		return geoERR_COMMON_NO_MEMORY;
	}
	memset(*ppDest,0,nSize);
	return geoRET_SUCC;
}

 void GEO_CALL FreeAndNull(void** ppDest)
{
	if (ppDest==geoNULL)
	{
		return;
	}
	if (*ppDest!=geoNULL)
	{
		free(*ppDest);
		*ppDest = geoNULL;
	}
}

 void GEO_CALL FreeStr(geoStr* pStr)
{
	FreeAndNull((void**)pStr);
}

 void GEO_CALL ClearStr(geoStr* pStr)
{
	if (*pStr!=geoNULL)
	{
		free(*pStr);
	}
	*pStr = geoNULL;
}

 void GEO_CALL ClearStr(geoStr pStr)
{
	ClearStr(&pStr);
}

 geoAPIStatus GEO_CALL NewAndCopyStr(geoStr* geotrDest,const geoChar * strSource)
{
	geoAPIStatus nRet = geoRET_SUCC;
	RET_ERR(NewMemory((geoVoid**)geotrDest,
		(geoUInt32)(strlen(strSource)+1)*sizeof(geoChar)));  /*�޸�ԭ��:����unique���������ռ䳤�Ȳ���*/
	strcpy(*geotrDest,strSource);
	return geoRET_SUCC;
}

 geoAPIStatus GEO_CALL NewAndCopyWStr(geoWStr* geotrDest,const geoWChar * strSource)
{
	geoAPIStatus nRet = geoRET_SUCC;
	RET_ERR(NewMemory((geoVoid**)geotrDest,
		(geoUInt32)(strlen((const geoChar *)strSource)+1)*sizeof(geoWChar)));  /*�޸�ԭ��:����unique���������ռ䳤�Ȳ���*/
	wcscpy((wchar_t *)*geotrDest,(const wchar_t *)strSource);
	return geoRET_SUCC;
}

 geoAPIStatus GEO_CALL geoStrRplace (geoStr* geotrDest,geoStr strSource)
{
	geoAPIStatus nRet = geoRET_SUCC;
	if ( IsNotNull(*geotrDest) )
	{
		ClearStr(geotrDest);
	}
	if (strSource)
	{
		RET_ERR(NewAndCopyStr(geotrDest,strSource));
	}
	return geoRET_SUCC;
}

 geoBool GEO_CALL IsVariantVariable(geoVariant* pVariant)
{
	return (pVariant->varType == geoVarTypeEnum::vTypeAString ||
		pVariant->varType == geoVarTypeEnum::vTypeWString ||
		pVariant->varType == geoVarTypeEnum::vTypeString ||
		pVariant->varType == geoVarTypeEnum::vTypeBlob);
}

 geoBool GEO_CALL IsVariantFixed(geoVariant* pVariant)
{
	return (!IsVariantVariable(pVariant));
}

 void GEO_CALL ClearVariant(geoVariant* pVariant)
{
	if (IsVariantVariable(pVariant) && (pVariant->vBlob.Length!=0))
	{
		free(pVariant->vBlob.Data);
	}
	memset(pVariant,0,sizeof(geoVariant));
}

 void GEO_CALL FreeVariant(geoVariant** ppVariant)
{
	if (ppVariant==geoNULL)
	{
		return;
	}
	if (*ppVariant==geoNULL)
	{
		return;
	}
	ClearVariant(*ppVariant);
	FreeAndNull((void**)ppVariant);
}





//////////////////////////////////////////////////////////////////////////
///����Ϊ�ڲ�����
//////////////////////////////////////////////////////////////////////////

static  geoBool  Bool_Equal(geoVariant *pFirst, geoVariant *pSecond)
{

	return ((pFirst->vBool) == (pSecond->vBool)) ? geoTRUE : geoFALSE;
}

static  geoBool  Int8_Equal(geoVariant *pFirst, geoVariant *pSecond)
{

	return ((pFirst->vInt8) == (pSecond->vInt8)) ? geoTRUE : geoFALSE;
}

static  geoBool  UInt8_Equal(geoVariant *pFirst, geoVariant *pSecond)
{
	return ((pFirst->vUInt8) == (pSecond->vUInt8)) ? geoTRUE : geoFALSE;
}

static  geoBool  Int16_Equal(geoVariant *pFirst, geoVariant *pSecond)
{
	return (pFirst->vInt16 == pSecond->vInt16) ? geoTRUE : geoFALSE;
}

static  geoBool  UInt16_Equal(geoVariant *pFirst, geoVariant *pSecond)
{
	return (pFirst->vUInt16 == pSecond->vUInt16) ? geoTRUE : geoFALSE;
}

static  geoBool  Int32_Equal(geoVariant*pFirst, geoVariant *pSecond)
{
	return (pFirst->vInt32==pSecond->vInt32) ? geoTRUE : geoFALSE;
}

static  geoBool  UInt32_Equal(geoVariant*pFirst, geoVariant *pSecond)
{
	return (pFirst->vUInt32 == pSecond->vUInt32) ? geoTRUE : geoFALSE;
}

static  geoBool  Int64_Equal(geoVariant*pFirst, geoVariant *pSecond)
{
	return (pFirst->vInt64==pSecond->vInt64) ? geoTRUE : geoFALSE;
}

static  geoBool  UInt64_Equal(geoVariant*pFirst, geoVariant *pSecond)
{
	return (pFirst->vUInt64==pSecond->vUInt64) ? geoTRUE : geoFALSE;
}

static  geoBool  Float_Equal(geoVariant*pFirst, geoVariant *pSecond)
{
	return fabs(pFirst->vFloat - pSecond->vFloat)<DOUBLE_EXCEPTION ? geoTRUE : geoFALSE;
}

static geoBool  Double_Equal(geoVariant*pFirst, geoVariant *pSecond)
{
	return fabs(pFirst->vDouble - pSecond->vDouble) <DOUBLE_EXCEPTION ? geoTRUE : geoFALSE;
}

static geoBool  AnsiString_Equal(geoVariant*pFirst, geoVariant *pSecond)
{
	return (pFirst->vAString.Length == pSecond->vAString.Length
		&& !strcmp(pFirst->vAString.Data,pSecond->vAString.Data)
		) ? geoTRUE : geoFALSE;
}

static geoBool  String_Equal(geoVariant*pFirst, geoVariant *pSecond)
{
	return(
		pFirst->vString.Length == pSecond->vString.Length
		&& !strcmp(pFirst->vString.Data,pSecond->vString.Data)
		) ? geoTRUE : geoFALSE;
}

static geoBool  UgeoodeString_Equal(geoVariant*pFirst, geoVariant *pSecond)
{

	return (
		pFirst->vWString.Length == pSecond->vWString.Length
		&&!wcscmp((wchar_t*)(pFirst->vWString.Data),
		(wchar_t*)(pSecond->vWString.Data))
		) ? geoTRUE : geoFALSE;

}

static geoBool  Time_Equal(geoVariant*pFirst, geoVariant *pSecond)
{

	return (pFirst->vTimeStamp == pSecond->vTimeStamp) ? geoTRUE : geoFALSE;
}
static geoBool  Blob_Equal(geoVariant *pFirst, geoVariant *pSecond)
{
	return (
		pFirst->vBlob.Length == pSecond->vBlob.Length
		&&memcmp(pFirst->vBlob.Data,pSecond->vBlob.Data,pFirst->vBlob.Length)
		) ? geoTRUE : geoFALSE;
}

// �˺����Ƿ�ֹת�Ʊ��г��ֿ�ָ��������ģ�û��ʵ�ʹ���
static geoBool  Equal_ErrorType(geoVariant *pFirst, geoVariant *pSecond)
{
	GEO_UNUSED_ARG(pFirst);
	GEO_UNUSED_ARG(pSecond);
	return geoFALSE;
}
//�˺����Ƿ�ֹת�Ʊ��г��ֿ�ָ��������ģ�û��ʵ�ʹ���
static geoAPIStatus changto_ErrorType(geoVariant* var)
{
	GEO_UNUSED_ARG(var);
	return geoERR_COMMON_DATATYPE;
}
static geoAPIStatus changto_Bool(geoVariant* var)
{
	geoBool tempbool=geoTRUE;
	switch (var->varType)
	{
	case geoVarTypeEnum::vTypeBool: 
		return geoRET_SUCC;
	case geoVarTypeEnum::vTypeInt8 :
		tempbool=var->vInt8 !=0;
		break;
	case geoVarTypeEnum::vTypeUInt8 :
		tempbool=var->vUInt8 !=0;
		break;
	case geoVarTypeEnum::vTypeInt16 :
		tempbool=var->vInt16!=0;
		break;
	case geoVarTypeEnum::vTypeUInt16 :
		tempbool=var->vUInt16!=0;
		break;
	case geoVarTypeEnum::vTypeInt32 :
		tempbool=var->vInt32!=0;
		break;
	case geoVarTypeEnum:: vTypeUInt32 :	
		tempbool=var->vUInt32!=0;
		break;
	case geoVarTypeEnum:: vTypeInt64 :
		tempbool=var->vInt64!=0;
		break;
	case geoVarTypeEnum:: vTypeUInt64 :	
		tempbool=var->vUInt64!=0;
		break;
	case geoVarTypeEnum:: vTypeFloat :
		tempbool=var->vFloat!=0;
		break;
	case geoVarTypeEnum:: vTypeDouble :	
		tempbool=var->vDouble!=0;
		break;
	case geoVarTypeEnum:: vTypeTimeStamp :
		tempbool=var->vTimeStamp !=0;
		break;
	case geoVarTypeEnum::vTypeAString :		
	case geoVarTypeEnum::vTypeWString :				
	case geoVarTypeEnum::vTypeString :						
	case geoVarTypeEnum::vTypeBlob :
	default:
		return geoERR_COMMON_DATACHANGE_FAILED;
	}
	var->vBool=tempbool;
	var->varType=geoVarTypeEnum::vTypeBool;
	return geoRET_SUCC;
}

static geoAPIStatus changto_Int8(geoVariant* var)
{
	geoInt8 tempint8=0;
	switch (var->varType)
	{
	case geoVarTypeEnum::vTypeBool: 
		tempint8=var->vBool;
		break;
	case geoVarTypeEnum::vTypeInt8 :
		tempint8=var->vInt8 ;
		break;
	case geoVarTypeEnum::vTypeUInt8 :
		if (var->vUInt8>SCHAR_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempint8=var->vUInt8;
		break;
	case geoVarTypeEnum::vTypeInt16 :
		if (var->vInt16<SCHAR_MIN||var->vInt16>SCHAR_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempint8=var->vInt16;
		break;
	case geoVarTypeEnum::vTypeUInt16 :
		if (var->vUInt16<SCHAR_MIN||var->vUInt16>SCHAR_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempint8=var->vUInt16;
		break;
	case geoVarTypeEnum::vTypeInt32 :
		if (var->vInt32<SCHAR_MIN||var->vInt32>SCHAR_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempint8=var->vInt32;
		break;
	case geoVarTypeEnum:: vTypeUInt32 :	
		if (var->vUInt32<SCHAR_MIN||var->vUInt32>SCHAR_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempint8=var->vUInt32;
		break;
	case geoVarTypeEnum:: vTypeInt64 :
		if (var->vInt64<SCHAR_MIN||var->vInt64>SCHAR_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempint8=var->vInt64;
		break;
	case geoVarTypeEnum:: vTypeUInt64 :	
		if (var->vUInt64<SCHAR_MIN||var->vUInt64>SCHAR_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempint8=var->vUInt64;
		break;
	case geoVarTypeEnum:: vTypeFloat :
		if (var->vFloat<SCHAR_MIN||var->vFloat>SCHAR_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempint8=var->vFloat;
		break;
	case geoVarTypeEnum:: vTypeDouble :
		if (var->vDouble<SCHAR_MIN||var->vDouble>SCHAR_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempint8=var->vDouble;
		break;
	case geoVarTypeEnum:: vTypeTimeStamp :
	case geoVarTypeEnum::vTypeAString :		
	case geoVarTypeEnum::vTypeWString :				
	case geoVarTypeEnum::vTypeString :						
	case geoVarTypeEnum::vTypeBlob :
	default:
		return geoERR_COMMON_DATACHANGE_FAILED;
	}
	var->vInt8=tempint8;
	var->varType=geoVarTypeEnum::vTypeInt8;
	return geoRET_SUCC;
}

static geoAPIStatus changto_UInt8(geoVariant* var)
{
	geoUInt8 tempint8=0;
	switch (var->varType)
	{
	case geoVarTypeEnum::vTypeBool: 
		tempint8=var->vBool;
		break;
	case geoVarTypeEnum::vTypeInt8 :
		if (var->vInt8<0)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempint8=var->vInt8;
		break;
	case geoVarTypeEnum::vTypeUInt8 :
		return geoRET_SUCC;
	case geoVarTypeEnum::vTypeInt16 :
		if (var->vInt16<0||var->vInt16>UCHAR_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempint8=var->vInt16;
		break;
	case geoVarTypeEnum::vTypeUInt16 :
		if (var->vUInt16>UCHAR_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempint8=var->vUInt16;
		break;
	case geoVarTypeEnum::vTypeInt32 :
		if (var->vInt32<0||var->vInt32>UCHAR_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempint8=var->vInt32;
		break;
	case geoVarTypeEnum:: vTypeUInt32 :	
		if (var->vUInt32>UCHAR_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempint8=var->vUInt32;
		break;
	case geoVarTypeEnum:: vTypeInt64 :
		if (var->vInt64<0||var->vInt64>UCHAR_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempint8=var->vInt64;
		break;
	case geoVarTypeEnum:: vTypeUInt64 :	
		if (var->vUInt64>UCHAR_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempint8=var->vUInt64;
		break;
	case geoVarTypeEnum:: vTypeFloat :
		if (var->vFloat<0||var->vFloat>UCHAR_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempint8=var->vFloat;
		break;
	case geoVarTypeEnum:: vTypeDouble :
		if (var->vDouble<0||var->vDouble>UCHAR_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempint8=var->vDouble;
		break;
	case geoVarTypeEnum:: vTypeTimeStamp :
	case geoVarTypeEnum::vTypeAString :		
	case geoVarTypeEnum::vTypeWString :				
	case geoVarTypeEnum::vTypeString :						
	case geoVarTypeEnum::vTypeBlob :
	default:
		return geoERR_COMMON_DATACHANGE_FAILED;
	}
	var->vUInt8=tempint8;
	var->varType=geoVarTypeEnum::vTypeUInt8;
	return geoRET_SUCC;
}
static geoAPIStatus changto_Int16(geoVariant* var)
{
	geoInt16 tempint16=0;
	switch (var->varType)
	{
	case geoVarTypeEnum::vTypeBool: 
		tempint16=var->vBool;
		break;
	case geoVarTypeEnum::vTypeInt8 :
		tempint16=var->vInt8 ;
		break;
	case geoVarTypeEnum::vTypeUInt8 :
		tempint16=var->vUInt8;
		break;
	case geoVarTypeEnum::vTypeInt16 :
		return geoRET_SUCC;
	case geoVarTypeEnum::vTypeUInt16 :
		if (var->vUInt16>SHRT_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempint16=var->vUInt16;
		break;
	case geoVarTypeEnum::vTypeInt32 :
		if (var->vInt32<SHRT_MIN||var->vInt32>SHRT_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempint16=var->vInt32;
		break;
	case geoVarTypeEnum:: vTypeUInt32 :	
		if (var->vUInt32>SHRT_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempint16=var->vUInt32;
		break;
	case geoVarTypeEnum:: vTypeInt64 :
		if (var->vInt64<SCHAR_MIN||var->vInt64>SCHAR_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempint16=var->vInt64;
		break;
	case geoVarTypeEnum:: vTypeUInt64 :	
		if (var->vUInt64<SHRT_MIN||var->vUInt64>SHRT_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempint16=var->vUInt64;
		break;
	case geoVarTypeEnum:: vTypeFloat :
		if (var->vFloat<SHRT_MIN||var->vFloat>SHRT_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempint16=var->vFloat;
		break;
	case geoVarTypeEnum:: vTypeDouble :
		if (var->vDouble<SHRT_MIN||var->vDouble>SHRT_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempint16=var->vDouble;
		break;
	case geoVarTypeEnum:: vTypeTimeStamp :
	case geoVarTypeEnum::vTypeAString :		
	case geoVarTypeEnum::vTypeWString :				
	case geoVarTypeEnum::vTypeString :						
	case geoVarTypeEnum::vTypeBlob :
	default:
		return geoERR_COMMON_DATACHANGE_FAILED;
	}
	var->vInt16=tempint16;
	var->varType=geoVarTypeEnum::vTypeInt16;
	return geoRET_SUCC;
}
static geoAPIStatus changto_UInt16(geoVariant* var)
{
	geoUInt16 tempuint16=0;
	switch (var->varType)
	{
	case geoVarTypeEnum::vTypeBool: 
		tempuint16=var->vBool;
		break;
	case geoVarTypeEnum::vTypeInt8 :
		if (var->vInt8<0)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempuint16=var->vInt8;
		break;
	case geoVarTypeEnum::vTypeUInt8 :
		tempuint16=var->vUInt8;
		break;
	case geoVarTypeEnum::vTypeInt16 :
		if (var->vInt16<0)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempuint16=var->vInt16;
		break;
	case geoVarTypeEnum::vTypeUInt16 :
		return geoRET_SUCC;
	case geoVarTypeEnum::vTypeInt32 :
		if (var->vInt32<0||var->vInt32>SHRT_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempuint16=var->vInt32;
		break;
	case geoVarTypeEnum:: vTypeUInt32 :	
		if (var->vUInt32>USHRT_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempuint16=var->vUInt32;
		break;
	case geoVarTypeEnum:: vTypeInt64 :
		if (var->vInt64<0||var->vInt64>USHRT_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempuint16=var->vInt64;
		break;
	case geoVarTypeEnum:: vTypeUInt64 :	
		if (var->vUInt64>USHRT_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempuint16=var->vUInt64;
		break;
	case geoVarTypeEnum:: vTypeFloat :
		if (var->vFloat<0||var->vFloat>USHRT_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempuint16=var->vFloat;
		break;
	case geoVarTypeEnum:: vTypeDouble :
		if (var->vDouble<0||var->vDouble>USHRT_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempuint16=var->vDouble;
		break;
	case geoVarTypeEnum:: vTypeTimeStamp :
	case geoVarTypeEnum::vTypeAString :		
	case geoVarTypeEnum::vTypeWString :				
	case geoVarTypeEnum::vTypeString :						
	case geoVarTypeEnum::vTypeBlob :
	default:
		return geoERR_COMMON_DATACHANGE_FAILED;
	}
	var->vUInt16=tempuint16;
	var->varType=geoVarTypeEnum::vTypeUInt16;
	return geoRET_SUCC;
}


static geoAPIStatus changto_Int32(geoVariant* var)
{
	geoInt32 tempint32=0;
	switch (var->varType)
	{
	case geoVarTypeEnum::vTypeBool: 
		tempint32=var->vBool;
		break;
	case geoVarTypeEnum::vTypeInt8 :
		tempint32=var->vInt8 ;
		break;
	case geoVarTypeEnum::vTypeUInt8 :
		tempint32=var->vUInt8;
		break;
	case geoVarTypeEnum::vTypeInt16 :
		tempint32=var->vInt16;
		break;
	case geoVarTypeEnum::vTypeUInt16 :
		tempint32=var->vUInt16;
		break;
	case geoVarTypeEnum::vTypeInt32 :
		return geoRET_SUCC;
	case geoVarTypeEnum:: vTypeUInt32 :	
		if (var->vUInt32>INT_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempint32=var->vUInt32;
		break;
	case geoVarTypeEnum:: vTypeInt64 :
		if (var->vInt64<INT_MIN||var->vInt64>INT_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempint32=var->vInt64;
		break;
	case geoVarTypeEnum:: vTypeUInt64 :	
		if (var->vUInt64>INT_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempint32=var->vUInt64;
		break;
	case geoVarTypeEnum:: vTypeFloat :
		if (var->vFloat<INT_MIN||var->vFloat>INT_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempint32=var->vFloat;
		break;
	case geoVarTypeEnum:: vTypeDouble :
		if (var->vDouble<INT_MIN||var->vDouble>INT_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempint32=var->vDouble;
		break;
	case geoVarTypeEnum:: vTypeTimeStamp :
	case geoVarTypeEnum::vTypeAString :		
	case geoVarTypeEnum::vTypeWString :				
	case geoVarTypeEnum::vTypeString :						
	case geoVarTypeEnum::vTypeBlob :
	default:
		return geoERR_COMMON_DATACHANGE_FAILED;
	}
	var->vInt32=tempint32;
	var->varType=geoVarTypeEnum::vTypeInt32;
	return geoRET_SUCC;
}
static geoAPIStatus changto_UInt32(geoVariant* var)
{
	geoUInt32 tempuint32=0;
	switch (var->varType)
	{
	case geoVarTypeEnum::vTypeBool: 
		tempuint32=var->vBool;
		break;
	case geoVarTypeEnum::vTypeInt8 :
		if (var->vInt8<0)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempuint32=var->vInt8;
		break;
	case geoVarTypeEnum::vTypeUInt8 :
		tempuint32=var->vUInt8;
		break;
	case geoVarTypeEnum::vTypeInt16 :
		if (var->vInt16<0)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempuint32=var->vInt16;
		break;
	case geoVarTypeEnum::vTypeUInt16 :
		tempuint32=var->vUInt16;
		break;
	case geoVarTypeEnum::vTypeInt32 :
		if (var->vInt32<0)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempuint32=var->vInt32;
		break;
	case geoVarTypeEnum:: vTypeUInt32 :	
		return geoRET_SUCC;
	case geoVarTypeEnum:: vTypeInt64 :
		if (var->vInt64<0||var->vInt64>UINT_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempuint32=var->vInt64;
		break;
	case geoVarTypeEnum:: vTypeUInt64 :	
		if (var->vUInt64>UINT_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempuint32=var->vUInt64;
		break;
	case geoVarTypeEnum:: vTypeFloat :
		if (var->vFloat<0||var->vFloat>UINT_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempuint32=var->vFloat;
		break;
	case geoVarTypeEnum:: vTypeDouble :
		if (var->vDouble<0||var->vDouble>UINT_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempuint32=var->vDouble;
		break;
	case geoVarTypeEnum:: vTypeTimeStamp :
	case geoVarTypeEnum::vTypeAString :		
	case geoVarTypeEnum::vTypeWString :				
	case geoVarTypeEnum::vTypeString :						
	case geoVarTypeEnum::vTypeBlob :
	default:
		return geoERR_COMMON_DATACHANGE_FAILED;
	}
	var->vUInt32=tempuint32;
	var->varType=geoVarTypeEnum:: vTypeUInt32;
	return geoRET_SUCC;
}

static geoAPIStatus changto_Int64(geoVariant* var)
{
	geoInt64 tempint64=0;
	switch (var->varType)
	{
	case geoVarTypeEnum::vTypeBool: 
		tempint64=var->vBool;
		break;
	case geoVarTypeEnum::vTypeInt8 :
		tempint64=var->vInt8 ;
		break;
	case geoVarTypeEnum::vTypeUInt8 :
		tempint64=var->vUInt8;
		break;
	case geoVarTypeEnum::vTypeInt16 :
		tempint64=var->vInt16;
		break;
	case geoVarTypeEnum::vTypeUInt16 :
		tempint64=var->vUInt16;
		break;
	case geoVarTypeEnum::vTypeInt32 :
		tempint64=var->vInt32;
		break;
	case geoVarTypeEnum:: vTypeUInt32 :	
		tempint64=var->vUInt32;
		break;
	case geoVarTypeEnum:: vTypeInt64 :
		return geoRET_SUCC;
	case geoVarTypeEnum:: vTypeUInt64 :	
		if (var->vUInt64>LLONG_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempint64=var->vUInt64;
		break;
	case geoVarTypeEnum:: vTypeFloat :
		if (var->vFloat<LLONG_MIN||var->vFloat>LLONG_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempint64=var->vFloat;
		break;
	case geoVarTypeEnum:: vTypeDouble :
		if (var->vDouble<LLONG_MIN||var->vDouble>LLONG_MAX)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempint64=var->vDouble;
		break;
	case geoVarTypeEnum:: vTypeTimeStamp :
	case geoVarTypeEnum::vTypeAString :		
	case geoVarTypeEnum::vTypeWString :				
	case geoVarTypeEnum::vTypeString :						
	case geoVarTypeEnum::vTypeBlob :
	default:
		return geoERR_COMMON_DATACHANGE_FAILED;
	}
	var->vInt64=tempint64;
	var->varType=geoVarTypeEnum:: vTypeInt64;
	return geoRET_SUCC;
}
static geoAPIStatus changto_UInt64(geoVariant* var)
{
	geoUInt64 tempuint64=0;
	switch (var->varType)
	{
	case geoVarTypeEnum::vTypeBool: 
		tempuint64=var->vBool;
		break;
	case geoVarTypeEnum::vTypeInt8 :
		if (var->vInt8<0)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempuint64=var->vInt8;
		break;
	case geoVarTypeEnum::vTypeUInt8 :
		tempuint64=var->vUInt8;
		break;
	case geoVarTypeEnum::vTypeInt16 :
		if (var->vInt16<0)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempuint64=var->vInt16;
		break;
	case geoVarTypeEnum::vTypeUInt16 :
		tempuint64=var->vUInt16;
		break;
	case geoVarTypeEnum::vTypeInt32 :
		if (var->vInt32<0)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempuint64=var->vInt32;
		break;
	case geoVarTypeEnum:: vTypeUInt32 :	
		tempuint64=var->vUInt32;
		break;
	case geoVarTypeEnum:: vTypeInt64 :
		if (var->vInt64<0)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempuint64=var->vInt64;
		break;
	case geoVarTypeEnum:: vTypeUInt64 :	
		return geoRET_SUCC;
	case geoVarTypeEnum:: vTypeFloat :
		if (var->vFloat<0)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempuint64=var->vFloat;
		break;
	case geoVarTypeEnum:: vTypeDouble :
		if (var->vDouble<0)
		{
			return geoERR_COMMON_DATACHANGE_FAILED;
		}
		tempuint64=var->vDouble;
		break;
	case geoVarTypeEnum:: vTypeTimeStamp :
	case geoVarTypeEnum::vTypeAString :		
	case geoVarTypeEnum::vTypeWString :				
	case geoVarTypeEnum::vTypeString :						
	case geoVarTypeEnum::vTypeBlob :
	default:
		return geoERR_COMMON_DATACHANGE_FAILED;
	}
	var->vUInt64=tempuint64;
	var->varType=geoVarTypeEnum:: vTypeUInt64;
	return geoRET_SUCC;
}


static geoAPIStatus changto_Float(geoVariant* var)
{
	geoFloat	 tempdata=0;
	switch (var->varType)
	{
	case geoVarTypeEnum::vTypeBool: 
		tempdata=var->vBool;
		break;
	case geoVarTypeEnum::vTypeInt8 :
		tempdata=var->vInt8;
		break;
	case geoVarTypeEnum::vTypeUInt8 :
		tempdata=var->vUInt8;
		break;
	case geoVarTypeEnum::vTypeInt16 :
		tempdata=var->vInt16;
		break;
	case geoVarTypeEnum::vTypeUInt16 :
		tempdata=var->vUInt16;
		break;
	case geoVarTypeEnum::vTypeInt32 :
		tempdata=var->vInt32;
		break;
	case geoVarTypeEnum:: vTypeUInt32 :	
		tempdata=var->vUInt32;
		break;
	case geoVarTypeEnum:: vTypeInt64 :
		tempdata=var->vInt64;
		break;
	case geoVarTypeEnum:: vTypeUInt64 :	
		tempdata=var->vUInt64;
		break;
	case geoVarTypeEnum:: vTypeFloat :
		return geoRET_SUCC;
	case geoVarTypeEnum:: vTypeDouble :
		tempdata=var->vDouble;
		break;
	case geoVarTypeEnum:: vTypeTimeStamp :
	case geoVarTypeEnum::vTypeAString :		
	case geoVarTypeEnum::vTypeWString :				
	case geoVarTypeEnum::vTypeString :						
	case geoVarTypeEnum::vTypeBlob :
	default:
		return geoERR_COMMON_DATACHANGE_FAILED;
	}
	var->vFloat=tempdata;
	var->varType=geoVarTypeEnum:: vTypeFloat;
	return geoRET_SUCC;
}
static geoAPIStatus changto_Double(geoVariant* var)
{
	geoFloat	 tempdata=0;
	switch (var->varType)
	{
	case geoVarTypeEnum::vTypeBool: 
		tempdata=var->vBool;
		break;
	case geoVarTypeEnum::vTypeInt8 :
		tempdata=var->vInt8;
		break;
	case geoVarTypeEnum::vTypeUInt8 :
		tempdata=var->vUInt8;
		break;
	case geoVarTypeEnum::vTypeInt16 :
		tempdata=var->vInt16;
		break;
	case geoVarTypeEnum::vTypeUInt16 :
		tempdata=var->vUInt16;
		break;
	case geoVarTypeEnum::vTypeInt32 :
		tempdata=var->vInt32;
		break;
	case geoVarTypeEnum:: vTypeUInt32 :	
		tempdata=var->vUInt32;
		break;
	case geoVarTypeEnum:: vTypeInt64 :
		tempdata=var->vInt64;
		break;
	case geoVarTypeEnum:: vTypeUInt64 :	
		tempdata=var->vUInt64;
		break;
	case geoVarTypeEnum:: vTypeFloat :
		return geoRET_SUCC;
	case geoVarTypeEnum:: vTypeDouble :
		tempdata=var->vDouble;
		break;
	case geoVarTypeEnum:: vTypeTimeStamp :
	case geoVarTypeEnum::vTypeAString :
	//	tempdata=(geoDouble)atof();
	case geoVarTypeEnum::vTypeWString :				
	case geoVarTypeEnum::vTypeString :						
	case geoVarTypeEnum::vTypeBlob :
	default:
		return geoERR_COMMON_DATACHANGE_FAILED;
	}
	var->vDouble=tempdata;
	var->varType=geoVarTypeEnum:: vTypeDouble;
	return geoRET_SUCC;
}

static geoAPIStatus changto_Time(geoVariant *var)
{
	geoTimeStamp	 tempdata=0;
	switch (var->varType)
	{
	case geoVarTypeEnum::vTypeBool: 
	case geoVarTypeEnum::vTypeInt8 :
	case geoVarTypeEnum::vTypeUInt8 :
	case geoVarTypeEnum::vTypeInt16 :
	case geoVarTypeEnum::vTypeUInt16 :
	case geoVarTypeEnum::vTypeInt32 :
	case geoVarTypeEnum:: vTypeUInt32 :	
	case geoVarTypeEnum:: vTypeInt64 :
	case geoVarTypeEnum:: vTypeUInt64 :	
	case geoVarTypeEnum:: vTypeFloat :
	case geoVarTypeEnum:: vTypeDouble :
		return geoERR_COMMON_DATACHANGE_FAILED;
	case geoVarTypeEnum:: vTypeTimeStamp :
		return geoRET_SUCC;
	case geoVarTypeEnum::vTypeAString :
	case geoVarTypeEnum::vTypeWString :				
	case geoVarTypeEnum::vTypeString :						
	case geoVarTypeEnum::vTypeBlob :
	default:
		return geoERR_COMMON_DATACHANGE_FAILED;
	}
// 	var->Time=tempdata;
// 	var->varType=geoVarTypeEnum:: vTypeTimeStamp;
	return geoRET_SUCC;	

}
static geoAPIStatus changto_AnsiString(geoVariant *var)
{
	geoChar temp[32]={'0'};
	struct tm * ptm=NULL;
	geoStr pch=NULL;
	int a=0;
	geoTimeStamp t = (var->vTimeStamp / 1000);
	switch (var->varType)
	{
	case geoVarTypeEnum::vTypeBool: 
		strcpy(temp,var->vBool==geoTRUE? "geoTRUE":"geoTRUE");
		break;
	case geoVarTypeEnum::vTypeInt8 :
		sprintf(temp,"%d",var->vUInt8);
		break;
	case geoVarTypeEnum::vTypeUInt8 :	
		sprintf(temp,"%d",var->vUInt8);
		break;
	case geoVarTypeEnum::vTypeInt16 :
		sprintf(temp,"%d",var->vInt16);
		break;
	case geoVarTypeEnum::vTypeUInt16 :
		sprintf(temp,"%d",var->vUInt16);
		break;
	case geoVarTypeEnum::vTypeInt32 :	
		sprintf(temp,"%d",var->vInt32);
		break;
	case geoVarTypeEnum:: vTypeUInt32 :
		sprintf(temp,"%d",var->vUInt32);
		break;

	case geoVarTypeEnum:: vTypeInt64 :
		sprintf(temp,"%lld",var->vInt64);
	case geoVarTypeEnum:: vTypeUInt64 :
		sprintf(temp,"%lld",var->vUInt64);
		break;
	case geoVarTypeEnum:: vTypeFloat :
	case geoVarTypeEnum:: vTypeDouble :
		sprintf(temp,"%f",var->vDouble);
		break;
	case geoVarTypeEnum:: vTypeTimeStamp :
		ptm=localtime((time_t*)(&t));
		sprintf(temp,"%04d-%02d-%02d %02d:%02d:%02d.%03d",ptm->tm_year+1900,ptm->tm_mon+1,ptm->tm_mday,ptm->tm_hour,ptm->tm_min,ptm->tm_sec,(geoUInt16)var->vTimeStamp % 1000);
		break;
	case geoVarTypeEnum::vTypeAString :
		return geoRET_SUCC;
	case geoVarTypeEnum::vTypeWString :	
	 a = wcslen((wchar_t*)var->vWString.Data);
		pch=(geoStr)alloca( a+1);
		for (int i=0;i<a;i++)
		{
			pch[i]=wctob(var->vWString.Data[i]);
		}
		Variant_AStrToVariant(pch,var);
		return geoRET_SUCC;
	case geoVarTypeEnum::vTypeString :
#ifdef ___UNICODE
		a = wcslen((wchar_t*)var->vWString.Data);
		pch=(geoStr)alloca( a+1);
		for (int i=0;i<a;i++)
		{
			pch[i]=wctob(var->vWString.Data[i]);
		}
		Variant_AStrToVariant(pch,var);
		return geoRET_SUCC;
#else if
		return geoRET_SUCC;
#endif
	case geoVarTypeEnum::vTypeBlob :
	default:
		return geoERR_COMMON_DATACHANGE_FAILED;
	}
	Variant_AStrToVariant(temp,var);
	return geoRET_SUCC;
}
static geoAPIStatus changto_UgeoodeString(geoVariant *var)
{
	geoWChar temp[32]={'0'};
	struct tm * ptm=NULL;
	geoWStr pch=NULL;
	int a=0;
	geoTimeStamp t = (var->vTimeStamp / 1000);
	switch (var->varType)
	{
#define TEMP (wchar_t*)  temp
	case geoVarTypeEnum::vTypeBool: 
		wcscpy(TEMP,var->vBool==geoTRUE? L"geoTRUE":L"geoFALSE");
		break;
	case geoVarTypeEnum::vTypeInt8 :
		swprintf(TEMP,L"%d",var->vUInt8);
		break;
	case geoVarTypeEnum::vTypeUInt8 :	
		swprintf(TEMP,L"%d",var->vUInt8);
		break;
	case geoVarTypeEnum::vTypeInt16 :
		swprintf(TEMP,L"%d",var->vInt16);
		break;
	case geoVarTypeEnum::vTypeUInt16 :
		swprintf(TEMP,L"%d",var->vUInt16);
		break;
	case geoVarTypeEnum::vTypeInt32 :	
		swprintf(TEMP,L"%d",var->vInt32);
		break;
	case geoVarTypeEnum:: vTypeUInt32 :
		swprintf(TEMP,L"%d",var->vUInt32);
		break;
	case geoVarTypeEnum:: vTypeInt64 :
		swprintf(TEMP,L"%lld",var->vInt64);
	case geoVarTypeEnum:: vTypeUInt64 :
		swprintf(TEMP,L"%lld",var->vUInt64);
		break;
	case geoVarTypeEnum:: vTypeFloat :
	case geoVarTypeEnum:: vTypeDouble :
		swprintf(TEMP,L"%f",var->vDouble);
		break;
	case geoVarTypeEnum:: vTypeTimeStamp :		
		ptm=localtime((time_t*)(&t));
		swprintf(TEMP,L"%04d-%02d-%02d %02d:%02d:%02d.%03d",ptm->tm_year+1900,ptm->tm_mon+1,ptm->tm_mday,ptm->tm_hour,ptm->tm_min,ptm->tm_sec, (geoUInt16)var->vTimeStamp % 1000);
		break;
	case geoVarTypeEnum::vTypeAString :
astr:	 a = wcslen((wchar_t*)var->vWString.Data);
		pch=(geoWStr)alloca( a+1);
		for (int i=0;i<a;i++)
		{
			pch[i]=btowc(var->vWString.Data[i]);
		}
		Variant_UStrToVariant(pch,var);		
	case geoVarTypeEnum::vTypeWString :	
		goto end;
	case geoVarTypeEnum::vTypeString :
#ifdef ___UNICODE
		goto end;
#endif
		goto astr;	
	case geoVarTypeEnum::vTypeBlob :
	default:
		return geoERR_COMMON_DATACHANGE_FAILED;
	}
	Variant_UStrToVariant(temp,var);
end:	return geoRET_SUCC;
}

static geoAPIStatus changto_String(geoVariant *var)
{
	geoAPIStatus nRet=geoRET_SUCC;
#ifdef ___UNICODE
	nRet =changto_UgeoodeString(var);
#else	
	nRet =changto_AnsiString(var);
#endif
	var->varType=geoVarTypeEnum::vTypeString;
	return nRet;
}
static geoAPIStatus  changto_Blob(geoVariant *var)
{
return  geoERR_COMMON_DATACHANGE_FAILED;
}

/// ���������ݵ�ͨ�õ������ڲ�ָ��ָ������ݿ�
/// ����ṹ���ڲ�ָ�벻�գ���ָ����ڴ潫���ͷ�
static  geoAPIStatus Copy_blockData_to_Variant ( geoVariant *pVariant,void *sdata,const unsigned length)
{
	char *p=NULL;
	char* PtempMem =(char*) malloc(length);
	if (NULL ==PtempMem)
	{
		return geoERR_COMMON_NO_MEMORY;
	}

	memcpy(PtempMem,sdata,length);

	p=pVariant->vAString.Data;  //����ָ��
	if (p)
	{       //������գ��ͷ��ڴ�
		free(p);
	}
	//ʹ�ڲ�ָ��ָ������
	pVariant->vAString.Data=PtempMem;
	return geoRET_SUCC;
}

//////////////////////////////////////////////////////////////////////////
/// �ڲ���������
//////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////
///����ת�Ʊ�
//////////////////////////////////////////////////////////////////////////

static PSVAEIANT_EQUAL  is_same_value[]=
{
	Equal_ErrorType,
	Bool_Equal,
	Int8_Equal,
	UInt8_Equal,
	Int16_Equal,
	UInt16_Equal,
	Int32_Equal,
	UInt32_Equal,
	Int64_Equal,
	UInt64_Equal,
	Float_Equal,
	Double_Equal,
	Time_Equal,
	String_Equal,
	AnsiString_Equal,
	UgeoodeString_Equal,
	Blob_Equal,
	Equal_ErrorType,
};

static VARIANT_CHANGETYPE Variant_changeType[]={
	changto_ErrorType,
	changto_Bool,
	changto_Int8,
	changto_UInt8,
	changto_Int16,
	changto_UInt16,
	changto_Int32,
	changto_UInt32,
	changto_Int64,
	changto_UInt64,
	changto_Float,
	changto_Double,
	changto_Time,
	changto_String,
	changto_AnsiString,
	changto_UgeoodeString,
	changto_Blob,
	changto_ErrorType,
};


///	 ΪgeoVariant��������ռ䲢��ʼ��,������������

geoAPIStatus GEO_CALL NEW_Variant(geoVariant **ppVariant)
{
    	//����ռ��ָ�룬����ֱ��ʹ�ö���ָ��
	geoVariant *pVariant = NULL;
	//����У��
	if (NULL == ppVariant)
	{
		return geoERR_COMMON_PARAMETER_INVALID;
	}

	pVariant = (geoVariant *) malloc( PSVARIANT_SIZE );
	if (NULL == pVariant)
	{
		return geoERR_COMMON_NO_MEMORY;
	}

	memset(pVariant,0,PSVARIANT_SIZE);
	*ppVariant = pVariant ;
	return geoRET_SUCC;
}

///		[API] ΪgeoVariant��������ռ䲢��ʼ��

/// 	����ǰ��geoVariant_New�������ͷ�*ppVariant��ָ��Ŀռ䣬
///		�����Ҫ�ͷţ�Ӧ�ȵ���geoVariant_Free������
///		���ú�DataType = nDataType , Value = 0(geoNULL) ;

geoAPIStatus   GEO_CALL Variant_New(PIN geoVarTypeEnum nDataType,
						 POUT geoVariant **ppVariant)
{
    geoAPIStatus ret = NEW_Variant(ppVariant);
	if (ret != geoRET_SUCC)
	{
	   return ret;
	}
	(*ppVariant)->varType= nDataType;

	return geoRET_SUCC;
}


///		�ͷ��������geoVariant�����ռ�
geoAPIStatus  GEO_CALL Variant_Free(PIN POUT geoVariant **ppVariant)
{
	if (NULL == ppVariant)
	{
		return geoERR_COMMON_PARAMETER_INVALID;
	}

	Variant_Clear(*ppVariant);    //����ڲ�����
	free(*ppVariant);             //�ͷ�variant�ṹ��
	*ppVariant=NULL;
	return geoRET_SUCC;
}


/// ���geoVariant����
/// ���ú�DataType���仯 , Value = 0(geoNULL)����Ҫʱ���ͷ�Value�ڴ�;
geoAPIStatus GEO_CALL Variant_Clear(PIN POUT geoVariant *pVariant)
{
	geoVarTypeEnum  type;
	if( geoNULL == pVariant )
	{
		return geoERR_COMMON_PARAMETER_INVALID;
	}


	type = pVariant->varType;
	if (geoVarTypeEnum::vTypeAString == type
		|| geoVarTypeEnum::vTypeWString == type
		|| geoVarTypeEnum::vTypeString == type
		|| geoVarTypeEnum::vTypeBlob == type)
	{
		char* pref = pVariant->vAString.Data;

		if (pref)
		{
			free(pref);
			pVariant->vAString.Data = NULL;
		}

	}
	memset(pVariant, 0, PSVARIANT_SIZE);
	pVariant->varType=geoVarTypeEnum::vTypeEmpty;
	return geoRET_SUCC;
}


///		���ַ�������geoVariant����

geoAPIStatus GEO_CALL Variant_StrToVariant(    PIN geoStr strStr,
								  PIN POUT geoVariant *pVariant)
{
	assert(pVariant!=geoNULL);
	Variant_Clear(pVariant);
	pVariant->varType = geoVarTypeEnum::vTypeString;
	pVariant->vString.Length = (geoUInt32)strlen(strStr);
	return NewAndCopyStr(&(pVariant->vString.Data),strStr);
}

///		��ANSI�ַ�������geoVariant����
geoAPIStatus GEO_CALL Variant_AStrToVariant(    PIN geoStr strAStr,
								  PIN POUT geoVariant *pVariant)
{
	assert(pVariant!=geoNULL);
	Variant_Clear(pVariant);
	pVariant->varType = geoVarTypeEnum::vTypeAString;
	pVariant->vAString.Length = (geoUInt32)strlen(strAStr);
	return NewAndCopyStr(&(pVariant->vAString.Data),strAStr);
}


///		��UNICODE�ַ�������geoVariant����
geoAPIStatus GEO_CALL Variant_UStrToVariant(PIN geoWStr strUStr,
								   PIN POUT geoVariant *pVariant)
{
	assert(pVariant!=geoNULL);
	Variant_Clear(pVariant);
	pVariant->varType = geoVarTypeEnum::vTypeWString;
	pVariant->vWString.Length = 
		(geoUInt32)wcslen((const wchar_t *)strUStr);
	return NewAndCopyWStr(&(pVariant->vWString.Data),strUStr);
}

///		�������ƴ�����geoVariant����
geoAPIStatus GEO_CALL Variant_BlobToVariant(	PIN geoByte *pByteList,
								   PIN geoUInt32 nLength,
								   PIN POUT geoVariant *pVariant)
{
	geoAPIStatus nRet = geoRET_SUCC;
	assert(pVariant!=geoNULL);
	Variant_Clear(pVariant);
	pVariant->varType = geoVarTypeEnum::vTypeBlob;
	pVariant->vBlob.Length = nLength;
	RET_ERR(NewMemory((geoVoid**)&(pVariant->vBlob.Data),nLength));
	memcpy(pVariant->vBlob.Data,pByteList,nLength);
	return geoRET_SUCC;
}

///		����geoVariant����
geoAPIStatus GEO_CALL Variant_Copy(		PIN geoVariant *pSource,
						  PIN POUT geoVariant *pDestination)
{
	geoVarTypeEnum type;
	if ( (NULL !=pSource) && (NULL !=pDestination))
	{
		Variant_Clear(pDestination);

		type=pSource->varType ;
		pDestination->varType = type;
		switch (type)
		{
		case geoVarTypeEnum::vTypeAString:
			pDestination->vAString.Length = pSource->vAString.Length;
			return Copy_blockData_to_Variant(
				pDestination,pSource->vAString.Data,
				pSource->vAString.Length+sizeof(geoChar));
		case geoVarTypeEnum::vTypeString:
			if (!(pSource->vString.Length))
			{
				memcpy(pDestination, pSource, PSVARIANT_SIZE);
				return geoRET_SUCC;
			}
			pDestination->vString.Length = pSource->vString.Length;
			return Copy_blockData_to_Variant(
				pDestination,pSource->vString.Data,
				pSource->vString.Length+sizeof(geoChar));
		case geoVarTypeEnum::vTypeWString:
			pDestination->vWString.Length = pSource->vWString.Length;
			return Copy_blockData_to_Variant(
				pDestination,pSource->vWString.Data,
				pSource->vWString.Length+sizeof(geoWChar));
		case geoVarTypeEnum::vTypeBlob:
			pDestination->vBlob.Length = pSource->vBlob.Length;
			return Copy_blockData_to_Variant(
				pDestination,pSource->vBlob.Data,
				pSource->vBlob.Length);
		default:
			memcpy(pDestination, pSource, PSVARIANT_SIZE);
		}
		return geoRET_SUCC;
	}
	return geoERR_COMMON_PARAMETER_INVALID;
}

geoBool GEO_CALL Variant_Equal(geoVariant *pFirst, geoVariant *pSecond)
{
	geoVarTypeEnum type;
	if (pFirst&&pSecond)
	{
		type=pFirst->varType;
		if (type == pSecond->varType)
		{
			return 	is_same_value[(geoUInt8)type](pFirst,pSecond);
		}
		return geoFALSE;

	}
	return geoFALSE;
}

///		��ȡgeoVariant����������ת����double���͵�ֵ
geoDouble GEO_CALL geoVariant_GetDouble( const geoVariant* value )
{
	switch (value->varType)
	{
	case geoVarTypeEnum::vTypeInt8:
		return static_cast<geoDouble> (value->vInt8);
	case geoVarTypeEnum::vTypeInt16:
		return static_cast<geoDouble> (value->vInt16);
	case geoVarTypeEnum::vTypeInt32:
		return static_cast<geoDouble> (value->vInt32);
	case geoVarTypeEnum:: vTypeInt64:
		return static_cast<geoDouble> (value->vInt64);
	case geoVarTypeEnum::vTypeUInt8:
		return static_cast<geoDouble> (value->vUInt8);
	case geoVarTypeEnum::vTypeUInt16:
		return static_cast<geoDouble> (value->vUInt16);
	case geoVarTypeEnum:: vTypeUInt32:
		return static_cast<geoDouble> (value->vUInt32);
	case geoVarTypeEnum:: vTypeUInt64:
		return static_cast<geoDouble> (value->vUInt64);
	case geoVarTypeEnum:: vTypeFloat:
		return static_cast<geoDouble> (value->vFloat);
	case geoVarTypeEnum:: vTypeDouble:
		return value->vDouble;
	case geoVarTypeEnum::vTypeBool:
		return  static_cast<geoDouble> (value->vBool);
	default:
		assert(0);
		return 0;
	}
}



geoAPIStatus GEO_CALL VariantTypeCast(geoVarTypeEnum DataType, geoVariant* var_src,geoVariant* var_dst )
{
	if (!(DataType > geoVarTypeEnum::vTypeEmpty && DataType < geoVarTypeEnum::vTypeMax))
	{
		return geoERR_COMMON_DATATYPE;
	}
	geoAPIStatus nRet =geoRET_SUCC;
	if (IsNotNull(var_dst))
	{
		RET_ERR(Variant_Clear(var_dst)); 
		RET_ERR(Variant_Copy(var_src,var_dst));
	}
	geoVariant* pvariant= IsNotNull(var_dst)? var_dst:var_src;
	return Variant_changeType[(geoUInt8)DataType](pvariant);
}
