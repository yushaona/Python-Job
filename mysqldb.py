import pymysql as pm
print("-----------用于数据库表压缩-----------")
try:
    dbName = str(input("请输入数据库名"))
    print(dbName)
    # 连接数据库
    db = pm.connect("115.28.139.39", "root", "y1y2g3j4fussen", "db_flybear", 2789)
    cursor = db.cursor()
    cursor.execute(
            " Select table_name from information_schema.tables "
            "where table_schema='%s' and row_format<>'Compressed' and engine='InnoDB' " %(dbName))
    if cursor.rowcount > 0:
        #有记录
        res = str(input("输入y开始执行"))
        if res.upper() == "Y":
            results = cursor.fetchall()
            #print(results)
            for row in results:
                print("处理表%s"%(row[0]))
                cursor.execute(" alter table %s.%s ROW_FORMAT=COMPRESSED KEY_BLOCK_SIZE=8 " %(dbName,row[0]))
        else:
            print("本次任务已放弃")
    else:
        print("无数据需要处理！")
except:
    print("发生未知异常错误")


db.close()


