

import os
from skimage import io,transform
import tensorflow as tf
import numpy as np
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from PIL import Image

def rgb2gray(im):
    return np.uint8(np.dot(im[...,:3], [0.299, 0.587, 0.114]))

folder="E:/downIm"
extension=".jpg"

def selectGray(fileList):
    grayList=[]
    num = len(fileList)
    for i in range(num):
        im=io.imread(fileList[i])
        if (im[:,:,1]==im[:,:,2]).all()&(im[:,:,2]==im[:,:,0]).all():
           grayList.extend([file_list[i]])
    return grayList
################ read file
file_list=[]
for path, subdirs, files in os.walk(folder):   #指定目录：data_base_dir中内容  
    for name in files:    
        if name.endswith(extension):      #文件以‘.jpg'，结尾  
           file_list.extend([os.path.join(path, name)])     #将jpg图片文件全部全部存入file_list列表中  
  
file_list= selectGray(file_list)
num = len(file_list)     #len(a):列表a长度  

inputSize=50
data=np.zeros([num,inputSize,inputSize])
for i in range(num):
    im=io.imread(file_list[i])
    im=transform.resize(rgb2gray(im), (inputSize, inputSize))
#    im = np.copy(im).astype('uint8')
#    io.imshow(im)
#    tempData.resize(1,500*500)
    data[i,:,:]=im[:,:]   

X = data.reshape([-1, inputSize, inputSize, 1])

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_samplewise_zero_center()
img_prep.add_samplewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_flip_updown ()
#img_aug.add_random_rotation(max_angle=3.)
img_aug.add_random_90degrees_rotation(rotations=[0,1,2,3])
img_aug.add_random_crop((inputSize,inputSize), 5)
img_aug.add_random_blur(sigma_max=1.0)


# Convolutional network building
network = input_data(shape=[None, inputSize, inputSize, 1],name='input',
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 4, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001,
                     name='target')

#
## Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.load('E:/Lisha/simpleConvGray.tflearn')



prd=model.predict(X)
label=np.argmax(prd,axis=1)
for i in range(num):
    if label[i]==0:
        head, tail = os.path.split(file_list[i])
        Image.open(file_list[i]).convert('RGB').save(os.path.join("E:/newTrain/gray/dentalCT", tail))
    if label[i]==1:
        head, tail = os.path.split(file_list[i])
        Image.open(file_list[i]).convert('RGB').save(os.path.join("E:/newTrain/gray/skullCT", tail))
    if label[i]==2:
        head, tail = os.path.split(file_list[i])
        Image.open(file_list[i]).convert('RGB').save(os.path.join("E:/newTrain/gray/toothCT", tail)) 
    if label[i]==3:
        head, tail = os.path.split(file_list[i])
        Image.open(file_list[i]).convert('RGB').save(os.path.join("E:/newTrain/gray/grayOther", tail))






    
        