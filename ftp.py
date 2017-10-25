import ftplib,socket
import logging
import  sys,time,os
import configparser as cp
import pymysql as py
import xlwings as xw
import datetime
from pymysql.cursors import SSCursor

#FTP数据上传和下载类
class FtpObject(object):
    def __init__(self,Ip,Port,userName,passWord):
        self.remoteIp = Ip
        self.remotePort = Port
        self.username = userName
        self.passwd = passWord
        self.ftp = None

    def uploadFile(self,remoteFileName,localFileName):
        if self.connectServer() == True:
            self.ftp.cwd(r"/jfx/")
            bufSize=2048
            fp = open(localFileName,'rb')
            self.ftp.storbinary('STOR '+remoteFileName,fp,bufSize)
            self.ftp.quit()
            return  True
        else:
            return  False

    def downloadFile(self,remoteFileName,localFileName):
        pass

    def connectServer(self):
        if self.ftp is None:
            logging.info(">>正在连接FTP...<<")
            try:
                self.ftp = ftplib.FTP()
                self.ftp.connect(self.remoteIp,self.remotePort)
                self.ftp.login(self.username,self.passwd)
                print(self.ftp.getwelcome())
                return True
            except socket.error:
                logging.info(">>远程FTP连接失败")
                return False
        return False

    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.ftp is not None:
            self.ftp.close()



#Excel操作类
class Excel(object):
    def __init__(self):
        self.app = None
        self.wb = None

    def loadData(self,**kargs):
        filename = kargs.get('filename', None)
        if filename is None:
            print("文件名未指定")
            return  -1
        try:
            self.wb = self.app.books.open(filename)
            rng = self.wb.sheets['sheet3'].range('A1').expand('table')
            num = rng.shape[1]
            print(num)
            #必须是两列
            if num == 2:
                result = rng.value
                print(result)
                return result
            else:
                return list()
        except:
            import sys
            data = sys.exc_info()
            print("loadData={}".format(data))

    def newExcel(self):
        self.wb = self.app.books.add()
        return  self.wb

    def __enter__(self):
        if self.app is None:
            self.app = xw.App(visible=False, add_book=False)
        return  self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.wb is not None:
            self.wb.close()
        self.app.quit()


def extractFileName(fullPath):
    return  fullPath[fullPath.rfind('\\')+1:len(fullPath)]

if __name__ == "__main__":
    curPath = os.getcwd()
    filePath = os.path.join(curPath, 'jfx')
    if os.path.exists(filePath) == False:
        os.mkdir(filePath)
    logPath = os.path.join(curPath, 'logs')
    if os.path.exists(logPath) == False:
        os.mkdir(logPath)
    pyName = extractFileName(sys.argv[0]).split('.')
    logPath = os.path.join(logPath, pyName[0] + ".log")
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
    conf.read("ftp.ini", encoding="utf8")
    db_host, db_port, db_user, db_pass, db_default = "115.28.139.39", 2789, "root", "y1y2g3j4fussen", "db_flybear"
    ip, port, username, password = '115.28.139.39', 1900, 'fussen', '1q2w3e'
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
            conf["ftp"] = {
                "ip": ip,
                "port": port,
                "username": username,
                "password": password
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
        with open('ftp.ini', 'w') as f:
            conf.write(f)

    logging.info("-----------数据库连接信息-----------")
    logging.info("host:%s" % (db_host))
    logging.info("port:%d" % (db_port))
    logging.info("user:%s" % (db_user))
    logging.info("password:%s" % (db_pass))
    logging.info("defaultDB:%s" % (db_default))
    logging.info('-----------ftp连接信息--------------')
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
    #ftp文件名
    fileName = "jfx_whitelist_" + time.strftime("%Y%m%d", time.localtime(time.time())) + ".xls"
    #本地文件路径
    filePath = os.path.join(filePath, fileName)
    logging.info('开始生成Excel文件-->>'+filePath)
    with conn.cursor(SSCursor) as cursor:
        with Excel() as newExcel:
            wb = newExcel.newExcel()
            sht = wb.sheets.active
            try:
                cursor.execute(
                    " select j.clinicname as '诊所名'," 
                    "j.doctorid as '申请医生ID'," 
                    "j.doctorname as '申请医生姓名'," 
                    "j.productname as '申请产品',"
                    "j.proposename  as '申请人',"
                    "j.proposemobile as '申请人电话',"
                    "j.total as '账单总额',"
                    "j.amount as '申请金额',"
                    "j.firstpay as '首付金额',"
                    "j.firstpayscale as '首付比例',"
                    "j.term as '期数',"
                    "j.notifytype as '通知类型',"
                    "j.applystatus as '通知状态',"
                    "j.remark as '备注信息',"
                    "j.orderno as '申请单编号',"
                    "j.orderid as '金服侠唯一编号',"
                    "j.cardid as '申请人身份证号' "
                    "from db_koala.t_jfx_apply j where j.datastatus=1 order by j.clinicuniqueid,j.createdate ")
                col_names = [col[0] for col in cursor.description]
                sht.range('A1').options(empty='NA', numbers=str).value = col_names
                sht.range('A1').expand('right').autofit()
                i = 2
                while True:
                    data = cursor.fetchone()
                    if data is None or not data:
                        break
                    cellIndex = 'A' + str(i)
                    sht.range(cellIndex).options(dates=datetime.date, empty='NA', numbers=str).value = data
                    #sht.range(cellIndex).expand('right').autofit()
                    i += 1
            except:
                import sys
                data = sys.exc_info()
                logging.error("导出Excel:{}".format(data))
            finally:
                #print(fileName)
                wb.save(filePath)
    logging.info('Excel文件生成完毕-->>'+filePath)
    # 上传文件到ftp
    if os.path.exists(filePath):
        logging.info('开始上传文件:')
        with  FtpObject("115.28.139.39", 1900, "fussen", "1q2w3e") as ftpObject:
            if ftpObject.uploadFile(fileName, filePath):
                logging.info('上传成功')
            else:
                logging.error('上传失败')
    else:
        logging.info('Excel文件不存在'+filePath)






