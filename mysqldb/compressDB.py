import  pymysql as py
from pymysql.cursors import SSCursor,SSDictCursor,Cursor,DictCursor
import  sys
from dbFunc.func import *
import configparser as cp

#读取配置文件
conf = cp.ConfigParser()
conf.read("upgradeDB.ini",encoding="utf8")
db_host,db_port,db_user,db_pass,db_default="127.0.0.1",2789,"root","y1y2g3j4fussen","db_flybear"
try:
    if conf.has_section("db") == False:
        conf["db"] = {"db_host":db_host,
                      "db_port":db_port,
                      "db_user":db_user,
                      "db_pass":db_pass,
                      "db_default":db_default}
        with open("upgradeDB.ini", "w") as upgradeDB:
            conf.write(upgradeDB)
    else:
        db_host = conf.get("db","db_host")
        db_port = conf.getint("db","db_port")
        db_user = conf.get("db","db_user")
        db_pass = conf.get("db","db_pass")
        db_default = conf.get("db","db_default")
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
print("host:%s" %(db_host))
print("port:%d"%(db_port))
print("user:%s"%(db_user))
print("password:%s"%(db_pass))
print("defaultDB:%s"%(db_default))


try:
    conn = py.connect(db_host, db_user, db_pass, db_default, db_port,charset='utf8')
    connNew = py.connect(db_host, db_user, db_pass, db_default, db_port,charset='utf8')
except:
    data = sys.exc_info()
    print("连接数据库错误:{}".format(data))
    exit(0)


print('''-----------数据库压缩脚本.py-----------)
     【操作指令】
     输入1-->>表示压缩库中[所有表]；-->>然后再输入【库】例如:db_koala 开始压缩
     输入2-->>表示压缩库中[指定表]；-->>然后再输入【库.表】例如:db_koala.t_image 开始压缩
     输入0-->>表示退出程序
''')

while True:
    try:
        res = int(input("请输入选项【0或1或2】"))
        if res == 0:
            print("退出程序")
            break
        elif res == 1:
            try:
                dbName = str(input("请输入【库】"))
                with conn.cursor(Cursor) as cursor:
                    compressDB(conn,cursor,dbName,connNew)
            except:
                import sys
                data = sys.exc_info()
                print("compressDB:{}".format(data))
                print("输入内容不合法res == 1")
        elif res == 2:
            try:
                name = str(input("请输入【库.表】"))
                pos = name.find(".")
                if pos == -1:
                    print("输入内容格式错误")
                else:
                    dbName = name[:pos]
                    tableName = name[pos+1:]
                    with conn.cursor(Cursor) as cursor:
                        compressTable(conn, cursor,dbName,tableName)
            except:
                print("输入内容不合法res == 2")
        else:
            pass
    except:
        pass

conn.close()
connNew.close()

