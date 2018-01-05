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

    # command line argument
    parser = argparse.ArgumentParser(description='Control program running argument')
    parser.add_argument('-n',"--num", type=int,default=0,help="Num is used to hash db_image")
    FLAGS = parser.parse_args()
    num = FLAGS.num
    if num < 0 or num > 3:
        logging.error("num is between [0,3]")
        exit(0)

    logging.info("Load model...")
    config_tf = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.InteractiveSession(config=config_tf)
    tl.ops.set_gpu_fraction(gpu_fraction=0.8)
    with tf.device('/cpu:0'):
        gnet = GoogleNetStructure(classes=4, batch_size=1, imagewidth=50,
                                  imageheight=50, imagechannel=1)
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
    bucket = utils.GetOssBucket('dt360')
    while True:
        try:
            logging.info("当前时间: %s " %(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
            now = time.time()
            if time.time() - now < 60:
                time.sleep(5)
            now = time.time()
            # hour = time.localtime(time.time()).tm_hour
            # if time not in [0,1,2,3,4,5,6,22,23]:
            #     time.sleep(60)
            #     continue

            #判断文件夹下未处理的文件数量
            # noSuckKey = []  # file not exist
            # lastNum = len(os.listdir(dirPath))
            # if lastNum > 100:
            #     logging.warning("待处理文件数 %d ,跳过下载..." %(lastNum))
            # else:
            #     logging.info("Download dcm...")
            #     begin, end = num * 50, num * 5 + 50
            #     for i in range(begin, end):
            #         with GetConnection().cursor(SSCursor) as cursor:
            #             cursor.execute(" select studyuid,seriesuid,sopuid from db_image.t_image_"+str(i)+" where isgray=-1 limit 5 ")
            #             while True:
            #                 data = cursor.fetchone()
            #                 if data is None or not data:
            #                     break
            #                 dcmPath = dirPath + '\\' + data[2] +'&t_image_'+ str(i) + '.dcm'
            #                 if os.path.exists(dcmPath) == False:
            #                     fileName = data[0] + '/' + data[1] + '/' + data[2] + '.dcm'
            #                     param = {}
            #                     param['funcid'] = capi.FUNC_ID_GET_OSSPATH
            #                     param['path'] = fileName
            #                     res = capi.FlybearCPlus(param)
            #                     osspath = res.get('osspath','')
            #                     if osspath == '':
            #                         logging.error('osspath计算失败, %s ' %(fileName))
            #                         continue
            #                     try:
            #                         bucket.get_object_to_file(osspath, dcmPath)
            #                         logging.info("dcm下载 >>> %s " % (dcmPath))
            #                     except oss2.exceptions.NoSuchKey:
            #                         logging.error('oss file not exist %s' %(osspath))
            #                         noSuckKey.append((data[2],"t_image_"+str(i)))
            #                         os.remove(dcmPath)
            #                     except:
            #                         data = sys.exc_info()
            #                         logging.error("Oss异常:{}".format(data))
            #                         os.remove(dcmPath)
            #                 # else:
            #                 #     logging.info("dcm >>> %s " %(dcmPath))
            #
            # suckKeyLen = len(noSuckKey)
            # if suckKeyLen > 0:
            #     try:
            #         logging.info("oss file not exist %d" %(suckKeyLen))
            #         for m in range(suckKeyLen):
            #             with GetConnection().cursor(SSCursor) as cursor:
            #                 cursor.execute(" update db_image." + noSuckKey[m][1] + " set isgray=2 where sopuid=%s",
            #                                (noSuckKey[m][0]))
            #     except:
            #         data = sys.exc_info()
            #         if 'Lost connection to MySQL' in data:
            #             GetConnection()
            #         logging.error("noSuchKey处理错误:{}".format(data))
            # try:
            #     logging.info("Search for *.dcm  in %s " %(dirPath))
            #     exts = ['dcm']
            #     dcm = os.walk(dirPath).__next__()[2]
            #     for dcmSample in dcm:
            #         if any(flag in  dcmSample for flag in exts):
            #             localPath = os.path.join(dirPath,dcmSample)
            #             logging.info("dcm处理>>> %s " % (localPath))
            #             paramx = {}
            #             paramx['funcid'] = capi.FUNC_ID_DICOM_GRAY
            #             paramx['src'] = localPath
            #             resx = capi.FlybearCPlus(paramx)
            #             if resx.get('code','0') == '1' and resx.get('isgray','-1') == '1':
            #                 jpgPath = utils.ChangeFileExt(localPath,'.jpg')
            #                 if os.path.exists(jpgPath) == False:
            #                     param = {}
            #                     param['funcid'] = capi.FUNC_ID_DICOM_TO_JPG
            #                     param['src'] = localPath
            #                     param['dest'] = jpgPath
            #                     res = capi.FlybearCPlus(param)
            #                     if res.get('code','0') == '1':
            #                         os.remove(localPath)
            #                     else:
            #                         logging.error("Dicom2Jpg failed, %s" %(localPath))
            #                         #os.remove(jpgPath)
            #                 else:
            #                     if GetFileSize(jpgPath) < 0.01:
            #                         os.remove(jpgPath)
            #                     else:
            #                         os.remove(localPath)
            #             elif resx.get('code','0') == '-100':
            #                 logging.error("Dicom2gray failed ,%s " %(localPath))
            #             else:
            #                 splitRes = dcmSample.split('&')
            #                 s = splitRes[1].split('.')
            #                 with GetConnection().cursor(SSCursor) as cursor:
            #                     cursor.execute(" update db_image."+s[0] +" set isgray=0 where sopuid=%s",(splitRes[0]))
            #                     os.remove(localPath)
            # except:
            #     data = sys.exc_info()
            #     if 'Lost connection to MySQL' in data:
            #         GetConnection()
            #     logging.error("dcm处理错误:{}".format(data))

            dbArgs,labels,samples=[],[],[]
            logging.info("Search for *.jpg in %s " %(dirPath))
            exts = ['jpg','jpeg']
            jpgs = os.walk(dirPath).__next__()[2]
            for jpgSample in jpgs:
                if any(flag in jpgSample for flag in exts):
                    samplePath = os.path.join(dirPath, jpgSample)
                    if GetFileSize(samplePath) < 0.01:
                        os.remove(samplePath)
                        continue
                    samples.append(samplePath)
                    splitRes = jpgSample.split('&')
                    imageName = splitRes[1].split('.')
                    dbArgs.append((splitRes[0],imageName[0]))
                    labels.append(0)
            total_num = len(samples)
            logging.info("JpgSamples number is %d " % (total_num))
            if total_num > 0:
                try:
                    logging.info('JpgSample is ok,convert image')
                    labels = np.asarray(labels, dtype='int32')
                    Y = np.zeros((len(labels), 4))
                    Y[np.arange(len(labels)), labels] = 1.
                    for i, s in enumerate(samples):
                        logging.info(s)
                        samples[i] = Imagefile.load_image(s)
                        samples[i] = Imagefile.resize_image(samples[i], 50, 50)
                        samples[i] = Imagefile.convert_color(samples[i], 'L')
                        samples[i] = Imagefile.pil_to_nparray(samples[i])
                        samples[i] /= 255.

                    samples = np.asarray(samples, dtype=np.float32)
                    samples = samples.reshape([-1, 50, 50, 1])
                    #result data
                    samples = np.asarray(samples, dtype=np.float32)
                    Y = np.asarray(Y, dtype=np.int64)
                    predictRes = gnet.predict(sess,samples,Y,isTest = False)
                    shape1 = predictRes.shape[0]
                    shape2 = predictRes.shape[1]
                    for i in range(shape1):
                        for j in range(shape2):
                            ss = predictRes[i, j]
                            r = 'GrayOther'
                            if ss == 0:
                                r = 'DentalCT'
                            if ss == 1:
                                r = 'SkullCT'
                            if ss == 2:
                                r = 'ToothCT'
                            if ss == 3:
                                r = 'GrayOther'
                            sopuid = dbArgs[int(i * gnet.batch_size + j)][0]
                            imageNum = dbArgs[int(i * gnet.batch_size + j)][1]
                            with GetConnection().cursor(SSCursor) as cursor:
                                cursor.execute(" update db_image." + imageNum + " set isgray=1,classes=%s where sopuid=%s", (r,sopuid))
                                os.remove(os.path.join(dirPath,sopuid+"&"+imageNum+".jpg"))
                except:
                    data = sys.exc_info()
                    if 'Lost connection to MySQL' in data:
                        GetConnection()
                    logging.error("jpg处理错误:{}".format(data))
        except:
            data = sys.exc_info()
            logging.error("系统错误:{}".format(data))
