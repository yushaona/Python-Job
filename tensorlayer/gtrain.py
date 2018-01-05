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
import os,time
from skimage import io,transform
import numpy as np
import googlenet
import utils
import tensorlayer as tl
import tensorflow as tf

extension=".jpg"
inputSize=50

def LoadFromDir(dir,datafile):
    if os.path.exists(datafile) == True:
        data = np.load(datafile)
    else:
        data = googlenet.LoadImageData(dir,extension,inputSize)
        np.save(datafile,data) 
    return data

dir = "E:/trainImg/gray/"
#data = googlenet.LoadImageData(dir,extension,inputSize)
destDir = utils.GetApplicationDir() + '\\TrainData\\'
utils.ForceDirectories(dir+'dentalCT')
Xdental = LoadFromDir(dir+'dentalCT', destDir + 'dental.npy')
Xskull = LoadFromDir(dir+'skullCT/side', destDir + 'skull.npy')
Xtooth = LoadFromDir(dir+'toothCT', destDir + 'tooth.npy')
Xother = LoadFromDir(dir+'grayOther', destDir + 'grayOther.npy')

Ydental = np.zeros([len(Xdental), 4])
Ydental[:,0]=1
Yskull = np.zeros([len(Xskull), 4])
Yskull[:,1]=1
Ytooth = np.zeros([len(Xtooth), 4])
Ytooth[:,2]=1
Yother = np.zeros([len(Xother), 4])
Yother[:,3]=1

x_train = np.row_stack([Xdental[0:3000,:,:], Xskull[0:900,:,:], Xtooth[0:3000,:,:], \
                  Xother[0:1000,:,:]]) 
y_train = np.row_stack([Ydental[0:3000,:], Yskull[0:900,:], Ytooth[0:3000,:], \
                  Yother[0:1000,:]]) 
x_train = x_train.reshape([-1, inputSize, inputSize, 1])

#图像分类的神经网络模型
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 50,50,1], name='x')
y_ = tf.placeholder(tf.int64, shape=[None,4], name='y_')
gnet = googlenet.PicClassModel(x,inputSize)
y = gnet.outputs
print(y_._shape)
#cost = tl.cost.cross_entropy(y,y_, 'cost')
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))  
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
y_op = tf.argmax(tf.nn.softmax(y), 1)
# define the optimizer
train_params = gnet.all_params
train_op = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999,epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)
# initialize all variables in the session
tl.layers.initialize_global_variables(sess)
# print network information
gnet.print_params()
gnet.print_layers()
# train the network
#tl.utils.fit(sess, gnet, train_op, cost, x_train, y_train, x, y_,
#acc=acc, batch_size=64, n_epoch=500, print_freq=5,
#X_val=x_train, y_val=y_train, eval_train=True)

batch_size=64
n_epoch=2
print_freq=5
X_val=None
y_val=None 
eval_train=True
network=gnet

print("Start training the network ...")
start_time_begin = time.time()
tensorboard_train_index, tensorboard_val_index = 0, 0
for epoch in range(n_epoch):
    start_time = time.time()
    loss_ep = 0; n_step = 0
    for X_train_a, y_train_a in tl.iterate.minibatches(x_train, y_train,
                                                batch_size, shuffle=True):
        feed_dict = {x: X_train_a, y_: y_train_a}
        feed_dict.update( network.all_drop )    # enable noise layers
        loss, step_acc,_ = sess.run([cost,acc, train_op], feed_dict=feed_dict)
        loss_ep += loss
        n_step += 1
        print("Epoch %d of %d step %d loss %f acc %f took %fs" % (epoch + 1, n_epoch,n_step,loss,step_acc,time.time() - start_time))
    loss_ep = loss_ep/ n_step

    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:       
        print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
        if eval_train is True:
            train_loss, train_acc, n_batch = 0, 0, 0
            for X_train_a, y_train_a in tl.iterate.minibatches(
                                    x_train, y_train, batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one( network.all_drop )    # disable noise layers
                feed_dict = {x: X_train_a, y_: y_train_a}
                feed_dict.update(dp_dict)
                if acc is not None:
                    err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                    train_acc += ac
                else:
                    err = sess.run(cost, feed_dict=feed_dict)
                train_loss += err;  n_batch += 1
            print("   train loss: %f" % (train_loss/ n_batch))
            if acc is not None:
                print("   train acc: %f" % (train_acc/ n_batch))
        if (X_val is not None) and (y_val is not None):
            val_loss, val_acc, n_batch = 0, 0, 0
            for X_val_a, y_val_a in tl.iterate.minibatches(
                                        X_val, y_val, batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one( network.all_drop )    # disable noise layers
                feed_dict = {x: X_val_a, y_: y_val_a}
                feed_dict.update(dp_dict)
                if acc is not None:
                    err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                    val_acc += acji
                else:
                    err = sess.run(cost, feed_dict=feed_dict)
                val_loss += err; n_batch += 1
            print("   val loss: %f" % (val_loss/ n_batch))
            if acc is not None:
                print("   val acc: %f" % (val_acc/ n_batch))
        else:
            print("Epoch %d of %d took %fs, loss %f" % (epoch + 1, n_epoch, time.time() - start_time, loss_ep))
print("Total training time: %fs" % (time.time() - start_time_begin))



tl.files.save_npz(network.all_params , name='gnet_model.npz')
sess.close()

