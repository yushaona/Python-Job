import os
import sys
import binascii
from django.db import connection
import utils,capi
from log import *

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "djan.settings")

def GetImageTableIndex(studyuid):
    multiplier1 = 31
    divider = 200
    value = 0
    for item in studyuid:  
        value = (value*multiplier1 + ord(item))%4294967296
    value = value%divider
    return str(value)

def GetCRC32(v):  
    """ 
    Generates the crc32 hash of the v. 按原c++的算法转过来，保证与原c++计算出来的结果一致
    @return: int, the str value for the crc32 of the v 
    """  
    s = ''
    i = 0
    length = len(v)
    for item in v:
        o = ord(item)
        x = o%256
        y = o//256
        s = s + chr(x) + chr(y)
        i = i + 2
        if i >= length:
            break 
    return binascii.crc32(str.encode(s))

def GetHashKey(uid):
    crc = GetCRC32(uid)
    hashCount = 4096
    res = crc%hashCount
    return res

def GetOSSPath(fileName):
    pos = fileName.find(':')
    if pos < 0:
        fileName = fileName.replace('\\','/')
        pos = fileName.find('/')
        if pos >= 0:
            val = fileName[0:pos]
    else:
        (filepath,val) = os.path.split(fileName)
    
    key = GetHashKey(val)
    x1 = key//64
    y1 = key%64
    multiplier1=31
    multiplier2=37
	#素数
    divider1 = 101
    divider2 = 103
    value = 0
    for item in val:
        value = (value*multiplier1 + ord(item))%4294967296 - 2147483648
    value %= divider1
    if value < 0: 
        value = value + divider1
    x2 = value
    value = 0
    for item in val:
        value = (value*multiplier2 + ord(item))%4294967296 - 2147483648
    value %= divider2
    if value < 0: 
        value = value + divider2
    y2 = value;
    path = str(x1) +"/"+ str(y1) + "/" + str(x2) + "/" + str(y2) + "/" + fileName
    return path

def DownImage(dir):
    #将图片下载到本地dir目录下
    uidDict = {}
    cursor=connection.cursor()
    cursor.execute('select studyuid from  db_koala.t_image order by updatetime desc limit 100 ')
    query = utils.FetchDict(cursor) #返回结果行游标直读向前，读取一条
    length = len(query)
    i = 0
    count = 0
    for row in query:
        studyuid = row['studyuid']
        tableName = 'db_image.t_image_' + GetImageTableIndex(studyuid)
        sql = 'select sopuid,seriesuid,studyuid from ' + tableName + ' where studyuid = %(studyuid)s '
        param = {'studyuid':studyuid}
        cursor.execute(sql, param)
        imageQuery = utils.FetchDict(cursor)
        for imageRow in imageQuery:
            #下载dicom图片到本地磁盘
            fileName = imageRow['studyuid'] + '/' + imageRow['seriesuid'] + '/' + imageRow['sopuid'] + '.dcm'
            bucket = utils.GetOssBucket('dt360-d')
            osspath = GetOSSPath(fileName)
            localPath = dir + '\\' + imageRow['sopuid'] + '.dcm'
            jpg = utils.ChangeFileExt(localPath,'.jpg')
            if os.path.exists(jpg):
                count = count + 1
                uidDict[imageRow['sopuid']] = imageRow
                continue
            try:
                bucket.get_object_to_file(osspath,localPath)
                #判断dicom是否灰度图像
                paramx={}
                paramx['funcid'] = capi.FUNC_ID_DICOM_GRAY
                paramx['src'] = localPath
                paramx['dest'] = jpg
                resx = capi.FlybearCPlus(paramx)
                if resx['isgray'] == 1:
                    param={}
                    param['funcid'] = capi.FUNC_ID_DICOM_TO_JPG
                    param['src'] = localPath
                    param['dest'] = jpg
                    res = capi.FlybearCPlus(param)
                    if res['code'] == '1':
                        os.remove(localPath)
                        count = count + 1
                        uidDict[imageRow['sopuid']] = imageRow
                else:
                    os.remove(localPath)
                    count = count + 1
                    uidDict[imageRow['sopuid']] = imageRow
                    i = i+1
            except Exception as ex:
                if os.path.exists(localPath):
                    os.remove(localPath)
                Log().error("Exception:" + str(ex))
        i = i+1
        if i % 10 == 0:
            Log().info('总检查数:' + str(length) + ",当前完成:" + str(i) )
    return uidDict


