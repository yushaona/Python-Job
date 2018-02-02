import  pymysql as py
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

print("-----------数据更新脚本.py-----------")
try:
    db = py.connect(db_host, db_user, db_pass, db_default, db_port,charset='utf8')
except:
    data = sys.exc_info()
    print("连接数据库错误:{}".format(data))
    exit(0)

try:
    res = str(input("是否确认运行:输入Y继续执行"))
    if res.upper() == "Y":        
        print("开始运行")
        cursor = db.cursor()
        # AddColumn(db,cursor,"db_koala","t_handleset","ItemID","VARCHAR(20)")
        # AddColumn(db, cursor, "db_koala", "t_handleset", "ItemName", "VARCHAR(64)")
        # AddColumn(db, cursor, "db_koala", "t_handlegroupmain", "IsDir", "TINYINT(3) DEFAULT 0 after HandleGroupName")
        # AddColumn(db, cursor, "db_koala", "t_handlegroupmain", "GroupMainIdentity", "VARCHAR(20) DEFAULT '' after IsDir")
        # AddColumn(db, cursor, "db_koala", "t_vipcard", "VipPassword", "varchar(45) DEFAULT ''")
        # CreateTable(db,cursor,"db_koala","t_plansettemplate",CreatePlanSetTemplate())
        # CreateTable(db, cursor, "db_koala", "t_qcitemtime", CreateQcItemTime())
        # CreateTable(db, cursor, "db_koala", "t_qcitemset", CreateQcItemSet())
        # CreateTable(db, cursor, "db_koala", "t_qcplan", CreateQcPlan())
        # CreateTable(db, cursor, "db_koala", "t_qcset", CreateQcSet())
        # AddColumn(db, cursor, "db_image", "t_image", "FileType",
        #           "VARCHAR(40) DEFAULT NULL COLLATE 'utf8_unicode_ci' COMMENT '文件类型' ")


        #UpdateHandle(db,cursor,"db_image","t_image")
        #ModifyColumn(db, cursor, "db_image", "t_image", "classes","VARCHAR(32) ")
        AddIndex(db, cursor, "db_image", "t_image")
        print("运行完毕")
    else:
        print("执行已取消")
except:
    db.rollback()
    print("发生未知异常错误")
db.close()



