""" GoogLeNet.
Applying 'GoogLeNet' to Oxford's 17 Category Flower Dataset classification task.
References:
    - Szegedy, Christian, et al.
    Going deeper with convolutions.
    - 17 Category Flower Dataset. Maria-Elena Nilsback and Andrew Zisserman.
Links:
    - [GoogLeNet Paper](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf)
    - [Flower Dataset (17)](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/)
"""

from __future__ import division, print_function, absolute_import
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import os
from skimage import io,transform
import numpy as np
import googlenet
from PIL import Image
import utils
from django.db import connection
import QueryImage
from log import *

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "djan.settings")

extension=".jpg"

def ClassifyImage(dir):
    #对dir中的jpg图像进行分类
    imageClass = {}
    inputSize=50
    model = googlenet.PicClassModel(inputSize)
    fileName = utils.GetApplicationDir() + '\\TrainData\\googlenet.model'
    model.load(fileName)

    data = googlenet.LoadImageData(dir, extension, inputSize)
    fileList = googlenet.ListFiles(dir,extension)
    num = len(fileList)
    if num == 0:
        return imageClass
    X = data.reshape([-1, inputSize, inputSize, 1])
    prd=model.predict(X)
    label=np.argmax(prd,axis=1)
    for i in range(num):
        head, tail = os.path.split(fileList[i])
        sopuid = utils.ChangeFileExt(tail,'')
        r = 'GrayOther'
        if label[i]==0:
            r = 'DentalCT'
        if label[i]==1:
            r = 'SkullCT'
        if label[i]==2: 
            r = 'ToothCT'
        if label[i]==3:
            r = 'GrayOther'
        imageClass[sopuid] = r
    return imageClass

dir = "E:/trainImg/input"
inputSize=50
uidDict = QueryImage.DownImage(dir)
imageClass = ClassifyImage(dir)
#将分类信息写入到数据库
cursor=utils.GetDBCursor()
for sopuid,iclass in imageClass.items(): 
    try:
        sql = 'select sopuid,class from  db_image.t_image_class where sopuid = %(sopuid)s'
        param = {'sopuid':sopuid}
        cursor.execute(sql, param)
        imageQuery = utils.FetchDict(cursor)
        if len(imageQuery) == 0:
            uid = uidDict.get(sopuid)
            if uid == None:
                continue
            sql = 'insert into db_image.t_image_class set studyuid = %(studyuid)s,seriesuid = %(seriesuid)s,sopuid = %(sopuid)s,class = %(class)s'
            param = {}
            param['studyuid'] = uid['studyuid']
            param['seriesuid'] = uid['seriesuid']
            param['sopuid'] = sopuid
            param['class'] = iclass
            cursor.execute(sql, param)
        else:
            dbClass = imageQuery[0]['class']
            if dbClass == iclass:
                continue
            sql = 'update db_image.t_image_class set class = %(class)s where sopuid = %(sopuid)s'
            param = {}
            param['sopuid'] = sopuid
            param['class'] = iclass
            cursor.execute(sql, param)
    except Exception as ex:
        Log().error(str(ex))

