
import os,sys
import urllib.request
from urllib import parse
import json
import pymysql as py
from pymysql.cursors import SSCursor
import argparse
import logging
import time


def reporthook(blocknum, blocksize, totalsize):
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r%5.1f%% %d / %d" % (
            percent, readsofar, totalsize)
        sys.stderr.write(s)
        if readsofar >= totalsize: # near the end
            sys.stderr.write("\n")
    else: # total size is unknown
        sys.stderr.write("read %d\n" % (readsofar,))


#一次下载一个图片,并存储到指定位置
def downImage(imageurl,storepath):
    count = 0
    while True:
        try:
            urllib.request.urlretrieve(imageurl,storepath)#下载成功,自动跳出
            return True
        except:
            count += 1
            if(count > 3):
                print('下载失败 %s' %(imageurl))
                return False

def extractFileName(fullPath):
    return  fullPath[fullPath.rfind('/')+1:len(fullPath)]



conn = None
#数据库连接参数
#db_host,db_port,db_user,db_pass,db_default = "rdsg2i6roz9n6uteej4up.mysql.rds.aliyuncs.com",3306,"dental360","fussenct2014","db_flybear"
db_host,db_port,db_user,db_pass,db_default = "115.28.139.39",2789,"root","y1y2g3j4fussen","db_flybear"

uri = 'http://139.129.85.215/image/WADO.php?action=LoadImage'
apiUri = 'http://139.129.204.124/service11/func.php'

# uri = 'http://115.28.139.39/image/WADO.php?action=LoadImage'
# apiUri = 'http://115.28.139.39/service11/func.php'
def GetConnection():
    global conn
    if conn is None:
        conn = py.connect(db_host, db_user, db_pass, db_default, db_port, charset='utf8', autocommit=True)
    else:
        conn.ping(True)
    return conn


if __name__ == "__main__":



    curPath = os.getcwd()
    # image path
    dirPath = os.path.join(curPath, r'TrainData\imagedata')
    if os.path.exists(dirPath) == False:
        os.mkdir(dirPath)

    for i in range(0,11):
        filePath = os.path.join(dirPath,str(i))
        if os.path.exists(filePath) == False:
            os.mkdir(filePath)

    # logging init
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

    parser = argparse.ArgumentParser(description='Control program running argument')
    parser.add_argument('-s', "--sleep", type=float, default=0.8, help="sleep time (s)")
    parser.add_argument('-n','--num',type=int,default=2,help='table hash value [0,3]')
    parser.add_argument('-c','--child',type=int,default=0,help='more num')
    FLAGS = parser.parse_args()
    sleeptime = FLAGS.sleep
    hashValue = FLAGS.num
    childNum = FLAGS.child

    if hashValue < 0 or hashValue > 3:
        logging.error('table hash value is between [0,3]')
        exit(0)

    logging.info("处理间隔 %3.1fs " % (sleeptime))
    logging.info("处理表 %d " % (hashValue))

    # logging.info("-----------数据库连接信息----------")
    # logging.info("host:%s" % (db_host))
    # logging.info("port:%d" % (db_port))
    # logging.info("userName:%s" % (db_user))
    # logging.info("password:%s" % (db_pass))
    # logging.info("defaultDB:%s" % (db_default))
    # try:
    #     isOk = str(input("数据库连接信息是否确认? Y继续执行,否则终止"))
    #     if isOk.upper() != "Y":
    #         logging.warning("manual terminal program")
    #         exit(0)
    # except ValueError as err:
    #     logging.error("{}".format(err))
    #     exit(0)
    # except:
    #     exit(0)

    # connect db
    try:
        GetConnection()
    except:
        data = sys.exc_info()
        logging.error("连接数据库错误:{}".format(data))
        exit(0)

    logging.info("Download jpg...")
    begin, end = hashValue * 50, hashValue * 50 + 50

    if childNum > 0:
        begin,end = begin+ (childNum -1) * 2,begin+ (childNum -1) * 2+2

    while True:
        for i in range(begin, end):
            succesList = ""
            tableName = 't_image_'+str(i)
            time.sleep(1)
            logging.info('查询'+tableName)

            param = [{"params": {"funcid": 6234, "table": tableName}}]
            param = parse.quote(json.dumps(param), encoding='utf-8')
            res = urllib.request.urlopen(urllib.request.Request(url='%s%s%s' % (apiUri, '?', 'param='+param))).read()
            obj = json.loads(res.decode(encoding='utf-8'))
            if obj[0]['code'] == '1':
                records = obj[0]['data']['records']
                for j in range(0, len(records)):
                    time.sleep(sleeptime)
                    imageurl = uri + '&StudyUID=' + records[j]['studyuid'] + '&SeriesUID=' + records[j]['seriesuid'] + '&SopUID=' + records[j]['sopuid'] + '&Columns=250&Rows=250'
                    classes = records[j]['classes']
                    fileNumber = 0
                    if classes == 'DentalCT':
                        fileNumber = 0
                    elif classes == 'SkullCT':
                        fileNumber = 1
                    elif classes == 'SkullFaceCT':
                        fileNumber = 2
                    elif classes == 'ToothCT':
                        fileNumber = 3
                    elif classes == 'CT':
                        fileNumber = 4
                    elif classes == 'EndoScopic':
                        fileNumber = 5
                    elif classes == 'IntraOral':
                        fileNumber = 6
                    elif classes == 'ExternOral':
                        fileNumber = 7
                    elif classes == 'Informed':
                        fileNumber = 8
                    elif classes == 'ModelPic':
                        fileNumber = 9
                    elif classes == 'GrayOther' or classes == 'RGBOther':
                        fileNumber = 10

                    if downImage(imageurl=imageurl,
                                 storepath=os.path.join(dirPath, str(fileNumber) + '\\' + records[j]['sopuid'] + '.jpg')) == True:
                        if succesList == "":
                            succesList = "'" + records[j]['sopuid'] + "'"
                        else:
                            succesList += ",'" + records[j]['sopuid'] + "'"

                        logging.info("下载成功t_image_" + str(i) + ">>" + records[j]['sopuid'])
                    else:
                        logging.error('下载失败t_image_' + str(i) + '>>' + records[j]['sopuid'])

            if succesList != "":
                param = [{"params": {"funcid": 6235, "table": tableName,'sopuids':succesList}}]
                param = parse.quote(json.dumps(param), encoding='utf-8')
                res = urllib.request.urlopen(urllib.request.Request(url='%s%s%s' % (apiUri, '?', 'param=' + param))).read()
                obj = json.loads(res.decode(encoding='utf-8'))
                if obj[0]['code'] != '1':
                    logging.error('更新数据失败')
