'''
先下载dcm图像到本地目录
遍历目录中dcm图像,分类为灰度和非灰度图
将灰度图像再分类
'''
import pymysql as py
import tensorflow as tf
import tensorlayer as tl
from pymysql.cursors import SSCursor
import argparse
from ImageTrain import  *
import logging
import sys,time,os
import utils,QueryImage
import capi
import Imagefile
import numpy as np
import oss2
def extractFileName(fullPath):
    return  fullPath[fullPath.rfind('/')+1:len(fullPath)]

conn = None
#数据库连接参数
db_host,db_port,db_user,db_pass,db_default = "rdsg2i6roz9n6uteej4up.mysql.rds.aliyuncs.com",3306,"dental360","fussenct2014","db_flybear"
#db_host,db_port,db_user,db_pass,db_default = "115.28.139.39",2789,"root","y1y2g3j4fussen","db_flybear"
def GetConnection():
    global conn
    if conn is None:
        conn = py.connect(db_host, db_user, db_pass, db_default, db_port, charset='utf8', autocommit=True)
    else:
        conn.ping(True)
    return conn

def GetFileSize(filePath):
    fsize = os.path.getsize(filePath)
    return round(fsize / float(1024),2)



imageHeight = 200
imageWidth = 200
imageChannel = 3
imageClasses = 11

if __name__ == "__main__":
    curPath = os.getcwd()
    #image path
    dirPath = os.path.join(curPath,'imagedata')
    if os.path.exists(dirPath) == False:
        os.mkdir(dirPath)
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
    parser.add_argument('-s', "--sleep", type=int, default=15, help="sleep time (s)")
    FLAGS = parser.parse_args()
    sleeptime = FLAGS.sleep
    logging.info("处理时间间隔 %ds " %(sleeptime))

    logging.info("-----------数据库连接信息----------")
    logging.info("host:%s" %(db_host))
    logging.info("port:%d" %(db_port))
    logging.info("userName:%s" % (db_user))
    logging.info("password:%s" % (db_pass))
    logging.info("defaultDB:%s" % (db_default))
    try:
        isOk = str(input("数据库连接信息是否确认? Y继续执行,否则终止"))
        if isOk.upper() != "Y":
            logging.warning("manual terminal program")
            exit(0)
    except ValueError as err:
        logging.error("{}".format(err))
        exit(0)
    except:
        exit(0)

    #connect db
    try:
        GetConnection()
    except:
        data = sys.exc_info()
        logging.error("连接数据库错误:{}".format(data))
        exit(0)

    logging.info("Load model...")
    config_tf = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.InteractiveSession(config=config_tf)
    tl.ops.set_gpu_fraction(gpu_fraction=0.8)
    with tf.device('/cpu:0'):
        gnet = GoogleNetStructure(classes=imageClasses, batch_size=1, imagewidth=imageWidth,
                                  imageheight=imageHeight, imagechannel=imageChannel)
        gnet.model()
        if tl.files.load_and_assign_npz(sess, name='model/model5.npz', network=gnet.network) == False:
            saver = tf.train.Saver()
            tl.layers.initialize_global_variables(sess)  # 变量的初始化
            ckpt = tf.train.get_checkpoint_state('./saver/')
            if ckpt and ckpt.model_checkpoint_path:
                logging.info("saver restore from %s" % (ckpt.model_checkpoint_path))
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                logging.error("model restore failed !")
                exit(0)
    now = time.time()
    while True:
        try:
            logging.info("当前时间: %s " %(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

            if time.time() - now < sleeptime:
                time.sleep(1)
                continue
            now = time.time()
            errArgs=[]
            dbArgs,labels,samples=[],[],[]
            logging.info("Search for *.jpg in %s " %(dirPath))
            exts = ['jpg','jpeg']
            #每次加载20张图片处理
            counter = 0
            jpgs = os.walk(dirPath).__next__()[2]
            for jpgSample in jpgs:
                if any(flag in jpgSample for flag in exts):
                    counter += 1
                    samplePath = os.path.join(dirPath, jpgSample)
                    splitRes = jpgSample.split('&')
                    imageName = splitRes[1].split('.')
                    if GetFileSize(samplePath) < 0.01:
                        errArgs.append((splitRes[0],imageName[0]))
                        if counter > 19:
                            break
                        else:
                            continue
                    samples.append(samplePath)
                    dbArgs.append((splitRes[0],imageName[0]))
                    labels.append(0)
                    if counter > 19:
                        break
            total_num = len(samples)
            logging.info("JpgSamples number is %d " % (total_num))
            if total_num > 0:
                try:
                    logging.info('JpgSample is ok,convert image')
                    labels = np.asarray(labels, dtype='int32')
                    Y = np.zeros((len(labels), imageClasses))
                    Y[np.arange(len(labels)), labels] = 1.
                    for i, s in enumerate(samples):
                        logging.info(s)
                        samples[i] = Imagefile.load_image(s)
                        samples[i] = Imagefile.resize_image(samples[i], imageWidth, imageHeight)
                        #samples[i] = Imagefile.convert_color(samples[i], 'L')
                        samples[i] = Imagefile.pil_to_nparray(samples[i])
                        samples[i] /= 255.

                    samples = np.asarray(samples, dtype=np.float32)
                    samples = samples.reshape([-1, imageWidth, imageHeight, imageChannel])
                    #result data
                    samples = np.asarray(samples, dtype=np.float32)
                    Y = np.asarray(Y, dtype=np.int64)
                    predictRes = gnet.predict(sess,samples,Y,isTest = False)
                    shape1 = predictRes.shape[0]
                    shape2 = predictRes.shape[1]
                    for i in range(shape1):
                        ss = predictRes[i, 0]
                        r = 'GrayOther'
                        if ss == 0:
                            r = 'DentalCT'
                        elif ss == 1:
                            r = 'SkullCT'
                        elif ss == 2:
                            r = 'SkullFaceCT'
                        elif ss == 3:
                            r = 'ToothCT'
                        elif ss == 4:
                            r = 'CT'
                        elif ss == 5:
                            r = 'EndoScopic'
                        elif ss == 6:
                            r = 'IntraOral'
                        elif ss == 7:
                            r = 'ExternOral'
                        elif ss == 8:
                            r = 'Informed'
                        elif ss == 9:
                            r = 'ModelPic'
                        elif ss == 10:
                            r = 'RGBOther'
                        sopuid = dbArgs[int(i * gnet.batch_size)][0]
                        imageNum = dbArgs[int(i * gnet.batch_size)][1]
                        with GetConnection().cursor(SSCursor) as cursor:
                            cursor.execute(" update db_image." + imageNum + " set ishandle=0,aiclasses=%s,classes=if(ismanual=1,classes,%s) where sopuid=%s ", (r,r,sopuid))
                            os.remove(os.path.join(dirPath,sopuid+"&"+imageNum+".jpg"))
                except:
                    data = sys.exc_info()
                    if 'Lost connection to MySQL' in data:
                        GetConnection()
                    logging.error("jpg处理错误:{}".format(data))
            errNum = len(errArgs)

            if errNum > 0:
                    logging.info("file is not ok  %d" %(errNum))
                    for m in range(errNum):
                        try:
                            with GetConnection().cursor(SSCursor) as cursor:
                                cursor.execute(" update db_image." + errArgs[m][1] + " set ishandle=-1 where sopuid=%s ",
                                               (errArgs[m][0]))
                                os.remove(os.path.join(dirPath, errArgs[m][0] + "&" + errArgs[m][1] + ".jpg"))
                        except:
                            data = sys.exc_info()
                            if 'Lost connection to MySQL' in data:
                                GetConnection()
                            logging.error("异常文件处理错误:{}".format(data))
        except:
            data = sys.exc_info()
            logging.error("系统错误:{}".format(data))
