import paramiko
import configparser as cp
import pymysql as py
import  sys,time,os
from pymysql.cursors import SSCursor
import logging

#CRITICAL > ERROR > WARNING > INFO > DEBUG > NOTSET

class PySFTP(object):
    def __init__(self,userName,passWord,ip,port):
        self.userName = userName
        self.passWord = passWord
        self.ip = ip
        self.port = port
        self.sftp = None
    def connect(self):
        if self.sftp is None:
            try:
                self.sftp = paramiko.Transport((self.ip,self.port))
            except Exception as e:
                logging.error("connect failed,error info: {}".format(e))
                return False
            else:
                try:
                    self.sftp.connect(username=self.userName,password=self.passWord)
                except Exception as e:
                    logging.error("login failed ,error info:{}".format(e))
                    return False
        return True

    #localFile 本地文件路径
    #remoteFile 远端文件路径
    def uploadFile(self,localFile,remoteFile):
        if self.connect() == True:
            try:
                sftp = paramiko.SFTPClient.from_transport(self.sftp)
                sftp.put(localFile,remoteFile)
            except Exception as e:
                logging.error("uploadFile error:{}".format(e))
                return False
            else:
                return True
        else:
            return False

    def  __enter__(self):
        return  self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.sftp is not None:
            self.sftp.close()


def extractFileName(fullPath):
    return  fullPath[fullPath.rfind('\\')+1:len(fullPath)]
    

if __name__ == "__main__":
    curPath = os.getcwd()
    filePath = os.path.join(curPath,'whitelist')
    if os.path.exists(filePath) == False:
        os.mkdir(filePath)
    logPath = os.path.join(curPath,'logs')
    if os.path.exists(logPath) == False:
        os.mkdir(logPath)
    pyName = extractFileName(sys.argv[0]).split('.')
    logPath = os.path.join(logPath,pyName[0]+".log")
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s %(asctime)s %(process)s %(thread)d %(threadName)s '
                               '%(filename)s %(funcName)s %(message)s',
                        datefmt='%Y-%m-%d %H-%M-%S',
                        filename=logPath,
                        filemode='a'
                        )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(message)s'))
    logging.getLogger('').addHandler(console)


    conf = cp.ConfigParser()
    conf.read("sftp.ini", encoding="utf8")
    db_host, db_port, db_user, db_pass, db_default = "127.0.0.1", 2789, "root", "y1y2g3j4fussen", "db_flybear"
    ip,port,username,password = 'outer.sftp.baidu.com',2222,'mdm_fs','qUsxmZDd2d5FYAY'
    IsWrite = False
    try:
        if conf.has_section("db") == False:
            conf["db"] = {"db_host": db_host,
                          "db_port": db_port,
                          "db_user": db_user,
                          "db_pass": db_pass,
                          "db_default": db_default}
            IsWrite = True

        else:
            db_host = conf.get("db", "db_host")
            db_port = conf.getint("db", "db_port")
            db_user = conf.get("db", "db_user")
            db_pass = conf.get("db", "db_pass")
            db_default = conf.get("db", "db_default")

        if conf.has_section("sftp") == False:
            conf["sftp"] = {
                "ip":ip,
                "port":port,
                "username":username,
                "password":password
            }
            IsWrite = True

    except cp.NoSectionError as err:
        logging.error("错误 {}".format(err))
        exit(0)
    except cp.NoOptionError as err:
        logging.error("错误 {}".format(err))
        exit(0)
    except:
        logging.error("读取ini异常错误")
        exit(0)

    if IsWrite:
        with open('sftp.ini','w') as f:
            conf.write(f)

    logging.info("-----------数据库连接信息-----------")
    logging.info("host:%s" % (db_host))
    logging.info("port:%d" % (db_port))
    logging.info("user:%s" % (db_user))
    logging.info("password:%s" % (db_pass))
    logging.info("defaultDB:%s" % (db_default))
    logging.info('-----------sftp连接信息--------------')
    logging.info("RemoteIp:{}".format(ip))
    logging.info("RemotePort:{}".format(port))
    logging.info("UserName:{}".format(username))
    logging.info("PassWord:{}".format(password))

    try:
        conn = py.connect(db_host, db_user, db_pass, db_default, db_port, charset='utf8')
    except:
        data = sys.exc_info()
        logging.error("连接数据库错误:{}".format(data))
        exit(0)

    fileName = "fussen_whitelist_"+time.strftime("%Y%m%d",time.localtime(time.time()))+".txt"

    filePath = os.path.join(filePath,fileName)

    with open(filePath,'w',encoding='utf8') as fp:
        with conn.cursor(SSCursor) as cursor:
            cursor.execute(" select applyername,idnumber,applyermobile,maxmoney,items,licensepic,cliniccreatetime,clinicname"
                           " from t_djd_apply_info where datastatus=1 and checkstatus=1 order by createtime ")
            while 1:
                row = cursor.fetchone()
                if not row or row is None:
                    break
                line = '\t'.join([str(d) for d in row])
                fp.write(line + "\n")

    #上传文件到百度sftp
    if os.path.exists(filePath):
        logging.info('开始上传文件:')
        with PySFTP("mdm_fs", "qUsxmZDd2d5FYAY", "outer.sftp.baidu.com", 2222) as o:
            remoteFile = '/'.join(('/data', fileName))
            if o.uploadFile(filePath, remoteFile):
                logging.info('上传成功')
            else:
                logging.error('上传失败')
