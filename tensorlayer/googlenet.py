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

import tensorflow as tf
import tensorlayer as tl
import os
from skimage import io, transform
import numpy as np
from log import *

def rgb2gray(im):
#rgb图转灰度图
    if im.ndim == 2:
        return im
    return np.uint8(np.dot(im[..., :3], [0.299, 0.587, 0.114]))

def ListFiles(dir,extension):
    file_list = []
    for path, subdirs, files in os.walk(dir):
        for name in files:
            if name.endswith(extension):
                #将jpg图片文件全部全部存入file_list列表
                file_list.extend([os.path.join(path, name)])
    return file_list

def LoadImageData(folder, extension, size):
#将folder中后缀名为extension的图片文件转成大小为size的正方形矩阵
    Log().info("getData, folder:" + folder)
    file_list = []
    for path, subdirs, files in os.walk(folder):
        for name in files:
            if name.endswith(extension):
                #将jpg图片文件全部全部存入file_list列表
                file_list.extend([os.path.join(path, name)])
    #len(a):列表a长度
    num = len(file_list)
    data = np.zeros([num, size, size,1])
    for i in range(0,num):
        im = io.imread(file_list[i])
        size0 = np.max(im.shape)
        scale = (size-1)/size0
        #print(i," , ",file_list[i],", ndim: " + str(im.ndim))
        x = rgb2gray(im)
        im = transform.rescale(x, scale)
        data[i, 0:im.shape[0], 0:im.shape[1],0] = im[:, :]
        if i%20 == 0:
            Log().info("getData, file index:" +  str(i) + ",total:" + str(num))
    return data

def PicClassModel(x,inputSize):
#图像分类的神经网络模型
    # Define the neural network structure
    with tf.variable_scope("googleNet_1"):
        network = tl.layers.InputLayer(x, name='input_layer')
        conv1_7_7 = tl.layers.Conv2dLayer(network,
                                        act = tf.nn.relu,
                                        shape = [7, 7, 1, 64],
                                        strides = [1,2,2,1],
                                        padding='SAME',
                                        name='conv1_7_7_s2')
        pool1_3_3 = tl.layers.PoolLayer(conv1_7_7,
                                        ksize=[1, 3, 3, 1],
                                        strides=[1, 2, 2, 1],
                                        padding='SAME',
                                        pool=tf.nn.max_pool,
                                        name='pool1_3_3_m')
        pool1_3_3 = tl.layers.LocalResponseNormLayer(pool1_3_3,depth_radius=5, bias=1.0,alpha=0.0001, beta=0.75,name='pool1_3_3')
        conv2_3_3_reduce = tl.layers.Conv2dLayer(pool1_3_3,
                                        act = tf.nn.relu,
                                        shape = [1, 1, 64, 1],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='conv2_3_3_reduce')

        conv2_3_3 = tl.layers.Conv2dLayer(conv2_3_3_reduce,
                                        act = tf.nn.relu,
                                        shape = [3, 3, 1, 192],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='conv2_3_3')
        norm2_3_3 = tl.layers.LocalResponseNormLayer(conv2_3_3,depth_radius=5, bias=1.0,alpha=0.0001, beta=0.75,name='norm2_3_3')
        pool2_3_3 = tl.layers.PoolLayer(conv2_3_3,
                                        ksize=[1, 3, 3, 1],
                                        strides=[1, 2, 2, 1],
                                        padding='SAME',
                                        pool=tf.nn.max_pool,
                                        name='pool2_3_3_s2')
        inception_3a_1_1 = tl.layers.Conv2dLayer(pool2_3_3,
                                        act = tf.nn.relu,
                                        shape = [1, 1, 192, 64],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_3a_1_1')
        inception_3a_3_3_reduce = tl.layers.Conv2dLayer(pool2_3_3,
                                        act = tf.nn.relu,
                                        shape = [1, 1, 192, 96],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_3a_3_3_reduce')
        inception_3a_3_3 = tl.layers.Conv2dLayer(inception_3a_3_3_reduce,
                                        act = tf.nn.relu,
                                        shape = [3,3,96,128],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_3a_3_3')
        inception_3a_5_5_reduce = tl.layers.Conv2dLayer(pool2_3_3,
                                        act = tf.nn.relu,
                                        shape = [16,16,192,1],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_3a_5_5_reduce')
        inception_3a_5_5 = tl.layers.Conv2dLayer(inception_3a_5_5_reduce,
                                        act = tf.nn.relu,
                                        shape = [32,32,1,5],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_3a_5_5')
        inception_3a_pool = tl.layers.PoolLayer(pool2_3_3,
                                        ksize=[1, 3, 3, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        pool=tf.nn.max_pool,
                                        name='inception_3a_pool')
        inception_3a_pool_1_1 = tl.layers.Conv2dLayer(inception_3a_pool,
                                        act = tf.nn.relu,
                                        shape = [32,32,192,1],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_3a_pool_1_1')
        # merge the inception_3a__
        inception_3a_output = tl.layers.ConcatLayer([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1],concat_dim=3)
        inception_3b_1_1 = tl.layers.Conv2dLayer(inception_3a_output,
                                        act = tf.nn.relu,
                                        shape = [128,128,198,1],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_3b_1_1')
        inception_3b_3_3_reduce = tl.layers.Conv2dLayer(inception_3a_output,
                                        act = tf.nn.relu,
                                        shape = [128,128,198,1],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_3b_3_3_reduce')
        inception_3b_3_3 = tl.layers.Conv2dLayer(inception_3b_3_3_reduce,
                                        act = tf.nn.relu,
                                        shape = [192,192,1,3],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_3b_3_3')
        inception_3b_5_5_reduce = tl.layers.Conv2dLayer(inception_3b_3_3_reduce,
                                        act = tf.nn.relu,
                                        shape = [32,32,1,1],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_3b_5_5_reduce')
        inception_3b_5_5 = tl.layers.Conv2dLayer(inception_3b_3_3_reduce,
                                        act = tf.nn.relu,
                                        shape = [96,96,1,5],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_3b_5_5')
        inception_3b_pool = tl.layers.PoolLayer(inception_3a_output,
                                        ksize=[1, 3, 3, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        pool=tf.nn.max_pool,
                                        name='inception_3b_pool')
        inception_3b_pool_1_1 = tl.layers.Conv2dLayer(inception_3b_pool,
                                        act = tf.nn.relu,
                                        shape = [64,64,198,1],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_3b_pool_1_1')
        #merge the inception_3b_*
        inception_3b_output = tl.layers.ConcatLayer([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1],concat_dim=3,name='inception_3b_output')
        pool3_3_3 = tl.layers.PoolLayer(inception_3b_output,
                                        ksize=[1, 3, 3, 1],
                                        strides=[1, 2, 2, 1],
                                        padding='SAME',
                                        pool=tf.nn.max_pool,
                                        name='pool3_3_3')
        inception_4a_1_1 = tl.layers.Conv2dLayer(pool3_3_3,
                                        act = tf.nn.relu,
                                        shape = [192,192,10,1],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_4a_1_1')
        inception_4a_3_3_reduce = tl.layers.Conv2dLayer(pool3_3_3,
                                        act = tf.nn.relu,
                                        shape = [96,96,10,1],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_4a_3_3_reduce')
        inception_4a_3_3 = tl.layers.Conv2dLayer(inception_4a_3_3_reduce,
                                        act = tf.nn.relu,
                                        shape = [208,208,1,3],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_4a_3_3')
        inception_4a_5_5_reduce = tl.layers.Conv2dLayer(pool3_3_3,
                                        act = tf.nn.relu,
                                        shape = [16,16,10,1],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_4a_5_5_reduce')
        inception_4a_5_5 = tl.layers.Conv2dLayer(inception_4a_5_5_reduce,
                                        act = tf.nn.relu,
                                        shape = [48,48,1,5],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_4a_5_5')
        inception_4a_pool = tl.layers.PoolLayer(pool3_3_3,
                                        ksize=[1, 3, 3, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        pool=tf.nn.max_pool,
                                        name='inception_4a_pool')
        inception_4a_pool_1_1 = tl.layers.Conv2dLayer(inception_4a_pool,
                                        act = tf.nn.relu,
                                        shape = [64,64,10,1],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_4a_pool_1_1')
        inception_4a_output = tl.layers.ConcatLayer([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1],concat_dim=3,name='inception_4a_output')
        inception_4b_1_1 = tl.layers.Conv2dLayer(inception_4a_output,
                                        act = tf.nn.relu,
                                        shape = [160,160,10,1],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_4b_1_1')
        inception_4b_3_3_reduce = tl.layers.Conv2dLayer(inception_4a_output,
                                        act = tf.nn.relu,
                                        shape = [112,112,10,1],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_4b_3_3_reduce')
        inception_4b_3_3 = tl.layers.Conv2dLayer(inception_4b_3_3_reduce,
                                        act = tf.nn.relu,
                                        shape = [224,224,1,3],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_4b_3_3')
        inception_4b_5_5_reduce = tl.layers.Conv2dLayer(inception_4a_output,
                                        act = tf.nn.relu,
                                        shape = [24,24,10,1],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_4b_5_5_reduce')
        inception_4b_5_5 = tl.layers.Conv2dLayer(inception_4b_5_5_reduce,
                                        act = tf.nn.relu,
                                        shape = [64,64,1,5],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_4b_5_5')
        inception_4b_pool = tl.layers.PoolLayer(inception_4a_output,
                                        ksize=[1, 3, 3, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        pool=tf.nn.max_pool,
                                        name='inception_4b_pool')
        inception_4b_pool_1_1 = tl.layers.Conv2dLayer(inception_4b_pool,
                                        act = tf.nn.relu,
                                        shape = [64,64,10,1],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_4b_pool_1_1')
        inception_4b_output = tl.layers.ConcatLayer([inception_4b_1_1, inception_4b_3_3, inception_4b_5_5, inception_4b_pool_1_1],concat_dim=3,name='inception_4b_output')
        inception_4c_1_1 = tl.layers.Conv2dLayer(inception_4b_output,
                                        act = tf.nn.relu,
                                        shape = [128,128,10,1],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_4c_1_1')
        inception_4c_3_3_reduce = tl.layers.Conv2dLayer(inception_4b_output,
                                        act = tf.nn.relu,
                                        shape = [128,128,10,1],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_4c_3_3_reduce')
        inception_4c_3_3 = tl.layers.Conv2dLayer(inception_4c_3_3_reduce,
                                        act = tf.nn.relu,
                                        shape = [256,256,1,3],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_4c_3_3')
        inception_4c_5_5_reduce = tl.layers.Conv2dLayer(inception_4b_output,
                                        act = tf.nn.relu,
                                        shape = [24,24,10,1],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_4c_5_5_reduce')
        inception_4c_5_5 = tl.layers.Conv2dLayer(inception_4c_5_5_reduce,
                                        act = tf.nn.relu,
                                        shape = [64,64,1,5],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_4c_5_5')
        inception_4c_pool = tl.layers.PoolLayer(inception_4b_output,
                                        ksize=[1, 3, 3, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        pool=tf.nn.max_pool,
                                        name='inception_4c_pool')
        inception_4c_pool_1_1 = tl.layers.Conv2dLayer(inception_4c_pool,
                                        act = tf.nn.relu,
                                        shape = [64,64,10,1],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_4c_pool_1_1')
        inception_4c_output = tl.layers.ConcatLayer([inception_4c_1_1, inception_4c_3_3, inception_4c_5_5, inception_4c_pool_1_1],concat_dim=3,name='inception_4c_output')
        inception_4d_1_1 = tl.layers.Conv2dLayer(inception_4c_output,
                                        act = tf.nn.relu,
                                        shape = [112,112,10,1],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_4d_1_1')
        inception_4d_3_3_reduce = tl.layers.Conv2dLayer(inception_4c_output,
                                        act = tf.nn.relu,
                                        shape = [144,144,10,1],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_4d_3_3_reduce')
        inception_4d_3_3 = tl.layers.Conv2dLayer(inception_4d_3_3_reduce,
                                        act = tf.nn.relu,
                                        shape = [288,288,1,3],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_4d_3_3')
        inception_4d_5_5_reduce = tl.layers.Conv2dLayer(inception_4c_output,
                                        act = tf.nn.relu,
                                        shape = [32,32,10,1],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_4d_5_5_reduce')
        inception_4d_5_5 = tl.layers.Conv2dLayer(inception_4d_5_5_reduce,
                                        act = tf.nn.relu,
                                        shape = [64,64,1,5],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_4d_5_5')
        inception_4d_pool = tl.layers.PoolLayer(inception_4c_output,
                                        ksize=[1, 3, 3, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        pool=tf.nn.max_pool,
                                        name='inception_4d_pool')
        inception_4d_pool_1_1 = tl.layers.Conv2dLayer(inception_4d_pool,
                                        act = tf.nn.relu,
                                        shape = [64,64,10,1],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_4d_pool_1_1')
        inception_4d_output = tl.layers.ConcatLayer([inception_4d_1_1, inception_4d_3_3, inception_4d_5_5, inception_4d_pool_1_1],concat_dim=3,name='inception_4d_output')
        inception_4e_1_1 = tl.layers.Conv2dLayer(inception_4d_output,
                                        act = tf.nn.relu,
                                        shape = [256,256,10,1],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_4e_1_1')
        inception_4e_3_3_reduce = tl.layers.Conv2dLayer(inception_4d_output,
                                        act = tf.nn.relu,
                                        shape = [160,160,10,1],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_4e_3_3_reduce')
        inception_4e_3_3 = tl.layers.Conv2dLayer(inception_4e_3_3_reduce,
                                        act = tf.nn.relu,
                                        shape = [320,320,1,3],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_4e_3_3')
        inception_4e_5_5_reduce = tl.layers.Conv2dLayer(inception_4d_output,
                                        act = tf.nn.relu,
                                        shape = [32,32,10,1],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_4e_5_5_reduce')
        inception_4e_5_5 = tl.layers.Conv2dLayer(inception_4e_5_5_reduce,
                                        act = tf.nn.relu,
                                        shape = [128,128,1,5],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_4e_5_5')
        inception_4e_pool = tl.layers.PoolLayer(inception_4d_output,
                                        ksize=[1, 3, 3, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        pool=tf.nn.max_pool,
                                        name='inception_4e_pool')
        inception_4e_pool_1_1 = tl.layers.Conv2dLayer(inception_4e_pool,
                                        act = tf.nn.relu,
                                        shape = [128,128,10,1],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_4e_pool_1_1')
        inception_4e_output = tl.layers.ConcatLayer([inception_4e_1_1, inception_4e_3_3, inception_4e_5_5, inception_4e_pool_1_1],concat_dim=3,name='inception_4e_output')
        pool4_3_3 = tl.layers.PoolLayer(inception_4e_output,
                                        ksize=[1, 3, 3, 1],
                                        strides=[1, 2, 2, 1],
                                        padding='SAME',
                                        pool=tf.nn.max_pool,
                                        name='pool4_3_3')
        inception_5a_1_1 = tl.layers.Conv2dLayer(pool4_3_3,
                                        act = tf.nn.relu,
                                        shape = [256,256,10,1],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_5a_1_1')
        inception_5a_3_3_reduce = tl.layers.Conv2dLayer(pool4_3_3,
                                        act = tf.nn.relu,
                                        shape = [160,160,10,1],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_5a_3_3_reduce')
        inception_5a_3_3 = tl.layers.Conv2dLayer(inception_5a_3_3_reduce,
                                        act = tf.nn.relu,
                                        shape = [320,320,1,3],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_5a_3_3')
        inception_5a_5_5_reduce = tl.layers.Conv2dLayer(pool4_3_3,
                                        act = tf.nn.relu,
                                        shape = [32,32,10,1],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_5a_5_5_reduce')
        inception_5a_5_5 = tl.layers.Conv2dLayer(inception_5a_5_5_reduce,
                                        act = tf.nn.relu,
                                        shape = [128,128,1,5],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_5a_5_5')
        inception_5a_pool = tl.layers.PoolLayer(pool4_3_3,
                                        ksize=[1, 3, 3, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        pool=tf.nn.max_pool,
                                        name='inception_5a_pool')
        inception_5a_pool_1_1 = tl.layers.Conv2dLayer(inception_5a_pool,
                                        act = tf.nn.relu,
                                        shape = [128,128,10,1],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_5a_pool_1_1')
        inception_5a_output = tl.layers.ConcatLayer([inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1],concat_dim=3,name='inception_5a_output')
        inception_5b_1_1 = tl.layers.Conv2dLayer(inception_5a_output,
                                        act = tf.nn.relu,
                                        shape = [384,384,10,1],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_5b_1_1')
        inception_5b_3_3_reduce = tl.layers.Conv2dLayer(inception_5a_output,
                                        act = tf.nn.relu,
                                        shape = [192,192,10,1],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_5b_3_3_reduce')
        inception_5b_3_3 = tl.layers.Conv2dLayer(inception_5b_3_3_reduce,
                                        act = tf.nn.relu,
                                        shape = [384,384,1,3],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_5b_3_3')
        inception_5b_5_5_reduce = tl.layers.Conv2dLayer(inception_5a_output,
                                        act = tf.nn.relu,
                                        shape = [48,48,10,1],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_5b_5_5_reduce')
        inception_5b_5_5 = tl.layers.Conv2dLayer(inception_5b_5_5_reduce,
                                        act = tf.nn.relu,
                                        shape = [128,128,1,5],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_5b_5_5')
        inception_5b_pool = tl.layers.PoolLayer(inception_5a_output,
                                        ksize=[1, 3, 3, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        pool=tf.nn.max_pool,
                                        name='inception_5b_pool')
        inception_5b_pool_1_1 = tl.layers.Conv2dLayer(inception_5b_pool,
                                        act = tf.nn.relu,
                                        shape = [128,128,10,1],
                                        strides = [1,1,1,1],
                                        padding='SAME',
                                        name='inception_5b_pool_1_1')

        inception_5b_output = tl.layers.ConcatLayer([inception_5b_1_1, inception_5b_3_3, inception_5b_5_5, inception_5b_pool_1_1],concat_dim=3,name='inception_5b_output')

        pool5_7_7 = tl.layers.PoolLayer(inception_5b_output,
                                        ksize=[1, 7, 7, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        pool=tf.nn.avg_pool,
                                        name='pool5_7_7')
        #防止CNN过拟合
        #当迭代次数增多的时候，可能出现网络对训练集拟合的很好（在训练集上loss很小），但是对验证集的拟合程度很差的情况
        gnet = tl.layers.DropoutLayer(pool5_7_7, keep =0.4,name='dropout5_7_7')
        print(gnet.outputs._shape)
        #全连接 负责对网络最终输出的特征进行分类预测，得出分类结果
        gnet = tl.layers.FlattenLayer(gnet, name='flatten_layer')
        print(gnet.outputs._shape)
        gnet = tl.layers.DenseLayer(gnet,n_units=4,act = tf.identity,name='output_layer')
        print(gnet.outputs._shape)
        return gnet