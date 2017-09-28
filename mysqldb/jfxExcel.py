# 用于金服侠Excel中数据导入

import xlwings as xw
from dbFunc.func import importEnteCode
import configparser as cp
import  pymysql as py
import  sys

class Excel(object):
    def __init__(self):
        print(1)
        self.app = None

    def loadData(self,**kargs):
        filename = kargs.get('filename', None)
        if filename is None:
            print("文件名未指定")
            return  -1
        try:
            wb = self.app.books.open(filename)
            rng = wb.sheets['sheet3'].range('A1').expand('table')
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
    def __enter__(self):
        print(2)
        if self.app is None:
            self.app = xw.App(visible=False, add_book=False)
        return  self
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(3)
        self.app.quit()

if __name__ == '__main__':

    conf = cp.ConfigParser()
    conf.read("upgradeDB.ini", encoding="utf8")
    db_host, db_port, db_user, db_pass, db_default = "127.0.0.1", 2789, "root", "y1y2g3j4fussen", "db_flybear"
    try:
        if conf.has_section("db") == False:
            conf["db"] = {"db_host": db_host,
                          "db_port": db_port,
                          "db_user": db_user,
                          "db_pass": db_pass,
                          "db_default": db_default}
            with open("upgradeDB.ini", "w") as upgradeDB:
                conf.write(upgradeDB)
        else:
            db_host = conf.get("db", "db_host")
            db_port = conf.getint("db", "db_port")
            db_user = conf.get("db", "db_user")
            db_pass = conf.get("db", "db_pass")
            db_default = conf.get("db", "db_default")
    except cp.NoSectionError as err:
        print("错误 {}".format(err))
        exit(0)
    except cp.NoOptionError as err:
        print("错误 {}".format(err))
        exit(0)
    except:
        print("读取ini异常错误")
        exit(0)

    print("-----------数据库连接信息-----------")
    print("host:%s" % (db_host))
    print("port:%d" % (db_port))
    print("user:%s" % (db_user))
    print("password:%s" % (db_pass))
    print("defaultDB:%s" % (db_default))

    try:
        conn = py.connect(db_host, db_user, db_pass, db_default, db_port, charset='utf8')
    except:
        data = sys.exc_info()
        print("连接数据库错误:{}".format(data))
        exit(0)

    with conn.cursor() as cursor:

        with Excel() as openExcel:
            allData = openExcel.loadData(filename='D:\\1.xls')
            if len(allData):
                for data in allData:
                    if isinstance(data,list):
                        dentalid = data[0]
                        entecode = data[1]
                        if dentalid is not None and entecode is not None:
                            if entecode.upper() == "NONE":
                                entecode = ''
                            importEnteCode(conn,cursor,dentalid,entecode)
