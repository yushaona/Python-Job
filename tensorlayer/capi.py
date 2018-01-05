import utils
import ctypes,json
from log import *
import sys
import logging

FUNC_ID_DICOM_TO_JPG				= 21136 #将dicom文件转换为jpg文件
FUNC_ID_DICOM_GRAY					= 21138 #判断dicom图像是否灰度图
FUNC_ID_GET_OSSPATH					= 21140 #获取OSSPath

strDllPath = utils.GetApplicationDir() + "\\FlyBearAPIX64.dll"
dll = None

def FlybearCPlus(dictVal):  
    """ 
    调用c++中FlybearAPI的Flybear_Execute_GO接口
    """  
    res = {}
    try:
        global dll
        if dll is None:
            dll = ctypes.WinDLL(strDllPath)
        txt = json.dumps(dictVal)
        dll.IFlyBear_Execute_PHP.argtypes = [ctypes.c_wchar_p]
        dll.IFlyBear_Execute_PHP.restype = ctypes.c_wchar_p
        sres = dll.IFlyBear_Execute_PHP(txt)  
        res = json.loads(sres)
    except:
        data = sys.exc_info()
        res['code'] = '-100'
        res['info'] = data
    return res


"""
param={}
param['funcid'] = FUNC_ID_DICOM_TO_JPG
param['src'] = 'N:\\trainImg\\input\\1.2.826.0.1.3680043.2.461.9245537.1373583235.dcm'
param['dest'] = 'N:\\trainImg\\input\\1.2.826.0.1.3680043.2.461.9245537.1373583235.jpg'
res = FlybearCPlus(param)
"""