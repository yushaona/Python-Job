'''
连接monogo数据库，删除诊所的相关数据，便于数据重新导入

PyMonogo接口文档
http://api.mongodb.com/python/current/api/pymongo/collection.html#pymongo.collection.Collection.delete_one

'''
from pymongo import  MongoClient
from pymongo.errors import ConnectionFailure,AutoReconnect
import  time

#db 数据库monogo连接
#collections 字符串列表
#theDict  文档条件

def ClearMonogo(db,collections,theDict):
    try:
        for i in collections:
            count = 0
            while True:
                try:
                    result = db[i].delete_many(theDict)
                    print(i + "删除数量" + str(result.deleted_count))
                    break
                except (AutoReconnect,ConnectionFailure):
                    count += 1
                    if count > 3:
                        print("ConnectionFailure,give up  " +i )
                        break

    except:
        import sys
        data = sys.exc_info()
        print("ClearMonogo:{}".format(data))



stdTokens = ('AppStdAdvPay_token','AppStdBillDoctPayFee_token','AppStdDoctBillFee_token','AppStdDoctDebts_token','AppStdDoctVisitCount_token','AppStdPayDoctPayFee_token','AppStdPayOut_token')
stdDatas = ('AppStdAdvPay','AppStdBillDoctPayFee','AppStdDoctBillFee','AppStdDoctDebts','AppStdDoctVisitCount','AppStdPayDoctPayFee','AppStdPayOut')
proTokens = ('AppProDoctPayFee_token','AppProDoctBillFee_token','AppProDoctDebts_token','AppProDoctDisCharge_token','AppProDoctVisitCount_token')
proDatas=('AppProDoctBillFee','AppProDoctDebts','AppProDoctDisCharge','AppProDoctPayFee','AppProDoctVisitCount')



if __name__ == "__main__":

    try:
        from urllib.parse import quote_plus  # @ / + 等特殊字符需要url编码
    except ImportError:
        from urllib import quote_plus

    username, password, host, port = "eagle", "wing2016", "115.28.139.39", 27017
    # 第一种连接方式
    conn = MongoClient(host=host,port=port,username=username,password=password,maxIdleTimeMS=10000)

    #第二种连接方式  URI的方式
    # monogoURI = "mongodb://%s:%s@%s:%s" % (quote_plus(username), quote_plus(password), host,27017)
    # conn = MongoClient(monogoURI)


    try:
        conn.admin.command('ismaster')
    except ConnectionFailure:
        print("Server not available")
        exit(0)

    #选择数据库
    db = conn.koala

    clinicid = str(input("请输入诊所ID"))

    tokenDict = {"clinicid":clinicid}
    dataDict = {"clinicuniqueid":clinicid}

    while True:
        try:
            res = int(input('''【0删除标准版】【1删除专业版】'''))
            if res == 0:#标准版
                try:
                    flag = int(input('''【0清空token】【1 清空token和data】'''))
                    if flag == 1:
                        ClearMonogo(db, stdTokens, tokenDict)
                        ClearMonogo(db,stdDatas,dataDict)
                    else:
                        ClearMonogo(db,stdTokens,tokenDict)
                except:
                    pass
            elif res ==1:#专业版
                try:
                    flag = int(input('''【0清空token】【1 清空token和data】'''))
                    if flag == 1:
                        ClearMonogo(db, proTokens, tokenDict)
                        ClearMonogo(db,proDatas,dataDict)
                    else:
                        ClearMonogo(db,proTokens,tokenDict)
                except:
                    pass
            else:
                print("输入数值错误！")
        except:
            pass