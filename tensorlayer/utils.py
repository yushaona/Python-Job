import os,sys,oss2,shutil
from django.db import connection

def GetFileExt(fileName): 
    return os.path.splitext(fileName)[1] 

def ChangeFileExt(fileName,ext):
    """修改文件的扩展名
    :param str fileName: 文件名
    :param str ext: 替换后的扩展名，带.号
    """
    ls = os.path.splitext(fileName)
    res = ls[0] + ext
    return res

def FetchDict(cursor):
    "将游标返回的结果保存到一个字典对象中"
    desc = cursor.description
    return [
    dict(zip([col[0] for col in desc], row))
    for row in cursor.fetchall()
    ]

def GetOssBucket(bucketName):
    auth = oss2.Auth('WlZWPVisjOXliOAs', 'LsVjn2JN2PoYZqJTshTWMas20IlrX1')
    endpoint = 'http://oss-cn-qingdao-internal.aliyuncs.com'
    bucket = oss2.Bucket(auth, endpoint, bucketName)
    return bucket

def RemoveDir(dir):
    shutil.rmtree(dir)

def ClearDir(dir):
    shutil.rmtree(dir)
    os.makedirs(dir)


#获取脚本文件的当前路径
def GetApplicationDir():
     #获取脚本路径
     path = sys.path[0]
     #判断为脚本文件还是py2exe编译后的文件，如果是脚本文件，则返回的是脚本的目录，如果是py2exe编译后的文件，则返回的是编译后的文件路径
     if os.path.isdir(path):
         return path
     elif os.path.isfile(path):
         return os.path.dirname(path)

def ForceDirectories(path):
    #创建一个新目录
    # 去除首位空格
    try:
        path = path.strip()
        # 去除尾部 \ 符号
        path=path.rstrip("\\")
        isExists=os.path.exists(path)
        if not isExists:
            os.makedirs(path)
            return True
    except Exception as ex:
        print(ex)
        return False

def isConnectionEnable():  
    try:  
        connection.connection.ping()
    except:  
        return False  
    else:  
        return True  
   
def GetDBCursor():  
    if isConnectionEnable() == False:  
        connection.close()  
    cursor=connection.cursor()
    return cursor
