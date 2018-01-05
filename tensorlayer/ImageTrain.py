'''
卷积神经网络 GoogleNet V1 模型

样本文件只能生成一个，并对单个样本文件进行训练
'''

from __future__ import division, print_function, absolute_import
import tensorflow as tf
import tensorlayer as tl
import  numpy as np
import os,time
import utils
import googlenet
import Inception_V1
import Imagefile
import argparse
import tflearn.datasets.mnist as mnist

#定义模型类
class GoogleNetStructure:
    def __init__(self,classes,batch_size=64,imagewidth=50,imageheight=50,imagechannel=1):
        self.network = None
        self.cost = None
        self.acc = None
        self.cost_avg = None
        self.cost_avg_op = None
        self.acc_avg = None
        self.acc_avg_op = None
        self.opt_op = None
        self.train_op = None
        self.cost_total = None
        self.y_op = None
        self.y_predict = None
        self.y_labels = None
        self.batch_size = batch_size
        self.imagewidth = imagewidth
        self.imageheight = imageheight
        self.channel = imagechannel
        self.classes = classes
        self.global_step = tf.Variable(0., name="Training_step",trainable=False)
        with tf.name_scope('model_place'):
            self.x = tf.placeholder(tf.float32, shape=[self.batch_size, self.imagewidth,self.imageheight,self.channel], name='x')
            self.y_ = tf.placeholder(tf.int64, shape=[self.batch_size,self.classes], name='y_')

    def model(self):
        self.network = Inception_V1.Model(self.x,self.classes)
        #self.network = googlenet.PicClassModel(self.x,self.inputSize)
        with tf.device('/gpu:0'):
            y = self.network.outputs
            self.y_predict = tf.argmax(tf.nn.softmax(y), 1)
            self.y_labels = tf.argmax(self.y_,1)

            ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=y, name='thecost'))
            #ce = tl.cost.cross_entropy(y, self.y_, name='cost_11')
            L2 = 0
            for p in tl.layers.get_variables_with_name('relu/W', True, True):
                L2 += tf.contrib.layers.l2_regularizer(0.004)(p)
            self.cost = ce + L2
            self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(y), 1),tf.argmax(self.y_,1) ), tf.float32),name='theacc')

            self.acc_avg = tf.train.ExponentialMovingAverage(0.9, self.global_step, name='moving_avg_1')
            self.acc_avg_op = self.acc_avg.apply([self.acc])
            self.cost_avg = tf.train.ExponentialMovingAverage(0.9, self.global_step, name='moving_avg_2')
            lss = [self.cost] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.cost_total = tf.add_n(lss, name="Total_Loss")
            self.cost_avg_op = self.cost_avg.apply([self.cost_total])

            train_params = self.network.all_params
            #self.opt_op = tf.train.MomentumOptimizer(learning_rate=0.0001,momentum=0.9).minimize(self.cost,var_list=train_params,global_step=self.global_step)
            self.opt_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999,epsilon=1e-08, use_locking=False).minimize(self.cost,var_list=train_params,global_step=self.global_step)
            with tf.control_dependencies([self.opt_op]):
                with tf.control_dependencies([self.acc_avg_op, self.cost_avg_op]):
                    self.train_op = tf.no_op(name='train_op')

    def predict(self,sess,x_data,y_data,isTest = True,batchSize=None):
        if self.y_predict is None:
            raise Exception(" GoogleNet y_predict is None")
        if self.y_labels is None:
            raise Exception(" GoogleNet y_labels is None")
        y = []
        if isTest:
            for X_a,y_a in tl.iterate.minibatches(x_data,y_data, self.batch_size, shuffle=False):
                feed_dict = {self.x: X_a,self.y_:y_a}
                feed_dict.update(self.network.all_drop)  # enable noise layers
                y.append(sess.run([self.y_predict,self.y_labels], feed_dict=feed_dict))
        else:

            for X_a,y_a in tl.iterate.minibatches(x_data,y_data, self.batch_size, shuffle=False):
                feed_dict = {self.x: X_a,self.y_:y_a}
                feed_dict.update(self.network.all_drop)  # enable noise layers
                y.append(sess.run(self.y_predict, feed_dict=feed_dict))
        y = np.asarray(y, dtype=np.int64)
        return y


    #模型的训练方法
    def fit(self,sess,
            TensorBoard,
            X_train,
            y_train,
            x,
            y_,
            print_freq=5,
            n_epoch=100,
            x_val=None, y_val=None,
            eval_freq=50,
            tensorboard_graph_vis=True):

        if self.network is None:
            raise Exception(' GoogleNet model is None ')
        if self.acc is None:
            raise Exception(' GoogleNet accuray is None')
        if self.cost is None:
            raise Exception(' GoogleNet cost is None')
        if self.opt_op is None:
            raise Exception(' GoogleNet opt_op is None')
        if self.train_op is None:
            raise Exception(' GoogleNet train_op is None')


        saver = tf.train.Saver()
        if TensorBoard:
            print("Create tensorboard event file...")
            # 定义日志存储目录
            tl.files.exists_or_mkdir('logs/')
            # 文件句柄
            train_writer = tf.summary.FileWriter('logs/train', sess.graph)
            val_writer = tf.summary.FileWriter('logs/validation', sess.graph)

            for param in self.network.all_params:
                #print('Param name ', param.name)
                tf.summary.histogram(param.name, param)
            # 添加损失函数和精度函数
            tf.summary.scalar('cost_avg', self.cost_avg.average(self.cost_total))
            tf.summary.scalar('cost', self.cost)
            tf.summary.scalar('acc_avg', self.acc_avg.average(self.acc))
            tf.summary.scalar('accuracy', self.acc)

            merged = tf.summary.merge_all()
            # Initalize all variables and summaries
            print("Finished! use $tensorboard --logdir=logs/ to start server")

        tl.layers.initialize_global_variables(sess)  # 变量的初始化
        ######### 判断是否已经存在正在训练中的参数模型##########
        ckpt = tf.train.get_checkpoint_state('./saver/')
        if ckpt and ckpt.model_checkpoint_path:
            print("saver restore from %s" % (ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)

        print("Start training the network ...")
        start_time_begin = time.time()
        tensorboard_val_index = 0
        for epoch in range(n_epoch):
            loss_ep = 0
            n_step = 0
            ####################训练数据,减少损失函数####################
            for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, self.batch_size, shuffle=True):
                start_time = time.time()
                #X_train_a = tl.prepro.threading_data(X_train_a, fn=self.distort_fn, is_train=True)
                feed_dict = {x: X_train_a, y_: y_train_a}
                feed_dict.update(self.network.all_drop)  # enable noise layers
                _ = sess.run([self.train_op], feed_dict=feed_dict)
                ts = time.time() - start_time
                n_step += 1

                feed_dict = {x: X_train_a, y_: y_train_a}
                dp_dict = tl.utils.dict_to_one(self.network.all_drop)  # disable noise layers
                feed_dict.update(dp_dict)
                result = sess.run(merged, feed_dict=feed_dict)
                train_writer.add_summary(result, self.global_step.eval(sess))


                if n_step % print_freq == 0 or n_step == 1:
                    dp_dict = tl.utils.dict_to_one(self.network.all_drop)  # disable noise layers
                    feed_dict = {x: X_train_a, y_: y_train_a}
                    feed_dict.update(dp_dict)
                    loss, acc = sess.run([self.cost, self.acc], feed_dict=feed_dict)
                    print("Epoch " + str(epoch) + " of " + str(n_epoch) + " step " + str(n_step) + ": Minibatch Loss= " + \
                          "{:.4f}".format(loss) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc) +" took time %fs " %(ts))

                # if self.global_step % tensorboard_step_freq == 0 or self.global_step == 1:
                #     self.tensor_step += 1
                #     dp_dict = tl.utils.dict_to_one(self.network.all_drop)  # disable noise layers
                #     feed_dict = {x: X_train_a, y_: y_train_a}
                #     feed_dict.update(dp_dict)
                #     result = sess.run(merged, feed_dict=feed_dict)
                #     train_writer.add_summary(result, self.global_step)
                if self.global_step.eval(sess) % 100 == 0:
                    saver.save(sess, './saver/model.ckpt', self.global_step)


                ########验证数据##########
                if self.global_step.eval(sess) % eval_freq == 0:
                    if (x_val is not None) and (y_val is not None):
                        print('evaluate model...')
                        for X_val_a, y_val_a in tl.iterate.minibatches(
                                            x_val, y_val, self.batch_size, shuffle=True):
                                        dp_dict = tl.utils.dict_to_one(self.network.all_drop)  # disable noise layers
                                        feed_dict = {x: X_val_a, y_: y_val_a}
                                        feed_dict.update(dp_dict)
                                        result = sess.run(merged, feed_dict=feed_dict)
                                        val_writer.add_summary(result, tensorboard_val_index)
                                        tensorboard_val_index += 1

            ############保存事件文件##############
            # if False:
            #     if epoch + 1 == 1 or (epoch + 1) % tensorboard_step_freq == 0:
            #         for X_train_a, y_train_a in tl.iterate.minibatches(
            #                 X_train, y_train, batch_size, shuffle=True):
            #             dp_dict = tl.utils.dict_to_one(self.network.all_drop)  # disable noise layers
            #             feed_dict = {x: X_train_a, y_: y_train_a}
            #             feed_dict.update(dp_dict)
            #             result = sess.run(merged, feed_dict=feed_dict)
            #             train_writer.add_summary(result, tensorboard_train_index)
            #             tensorboard_train_index += 1
            #         if (x_val is not None) and (y_val is not None):
            #             for X_val_a, y_val_a in tl.iterate.minibatches(
            #                     x_val, y_val, batch_size, shuffle=True):
            #                 dp_dict = tl.utils.dict_to_one(self.network.all_drop)  # disable noise layers
            #                 feed_dict = {x: X_val_a, y_: y_val_a}
            #                 feed_dict.update(dp_dict)
            #                 result = sess.run(merged, feed_dict=feed_dict)
            #                 val_writer.add_summary(result, tensorboard_val_index)
            #                 tensorboard_val_index += 1

            ##########判断日志打印频率###########
            # if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            #     if (x_val is not None) and (y_val is not None):
            #         print("Epoch %d of %d took time %fs" % (epoch + 1, n_epoch, used_time))
            #         ##########间隔print_freq 层，检查一次训练数据的平均损失度和平均精度
            #         if eval_train is True:
            #             train_loss, train_acc, n_batch = 0, 0, 0
            #             for X_train_a, y_train_a in tl.iterate.minibatches(
            #                     X_train, y_train, batch_size, shuffle=True):
            #                 dp_dict = tl.utils.dict_to_one(self.network.all_drop)  # disable noise layers
            #                 feed_dict = {x: X_train_a, y_: y_train_a}
            #                 feed_dict.update(dp_dict)
            #                 if self.acc is not None:
            #                     err, ac = sess.run([self.cost, self.acc], feed_dict=feed_dict)
            #                     train_acc += ac
            #                 else:
            #                     err = sess.run(self.cost, feed_dict=feed_dict)
            #                 train_loss += err
            #                 n_batch += 1
            #             print("   train loss: %f" % (train_loss / n_batch))
            #             if self.acc is not None:
            #                 print("   train acc: %f" % (train_acc / n_batch))
            #
            #         ########### 用校验数据，检查一次损失度########
            #         val_loss, val_acc, n_batch = 0, 0, 0
            #         for X_val_a, y_val_a in tl.iterate.minibatches(
            #                 x_val, y_val, batch_size, shuffle=True):
            #             dp_dict = tl.utils.dict_to_one(self.network.all_drop)  # disable noise layers
            #             feed_dict = {x: X_val_a, y_: y_val_a}
            #             feed_dict.update(dp_dict)
            #             if self.acc is not None:
            #                 err, ac = sess.run([self.cost, self.acc], feed_dict=feed_dict)
            #                 val_acc += ac
            #             else:
            #                 err = sess.run(self.cost, feed_dict=feed_dict)
            #             val_loss += err;
            #             n_batch += 1
            #         print("   val loss: %f" % (val_loss / n_batch))
            #         if self.acc is not None:
            #             print("   val acc: %f" % (val_acc / n_batch))
            #     else:
            #         print("Epoch %d of %d took %fs, loss %f" % (epoch + 1, n_epoch, used_time, loss_ep))
        print("Total training time: %fs" % (time.time() - start_time_begin))
        saver.save(sess,'./saver/model.ckpt')



if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description='Control program running argument')
        parser.add_argument('--istrain', type=int, default=0,help='If istrain=1 then train data else predict data')
        FLAGS = parser.parse_args()
        TrainAndSave = FLAGS.istrain
        print("TrainAndSave is %d" %(TrainAndSave))

        X_train,y_train,X_test,y_test= Imagefile.load_data(dirname='TrainData/',imagefolder='imagedata',convert_gray=False,one_hot=True,resize_pics=(200, 200))
        print('X_train.shape', X_train.shape)
        print('y_train.shape', y_train.shape)
        print('X_test.shape', X_test.shape)
        print('y_test.shape', y_test.shape)


        # X_train, y_train, X_val, y_val, X_test, y_test = \
        #     tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1),path=r'MNIST_data/')
        #
        # X_train = np.asarray(X_train, dtype=np.float32)
        # y_train = Imagefile.to_categorical(y_train,np.max(y_train)+1)
        # y_train = np.asarray(y_train, dtype=np.int64)
        # X_val = np.asarray(X_val, dtype=np.float32)
        # y_val = Imagefile.to_categorical(y_val,np.max(y_val)+1)
        # y_val = np.asarray(y_val, dtype=np.int64)
        # X_test = np.asarray(X_test, dtype=np.float32)
        # y_test = Imagefile.to_categorical(y_test, np.max(y_test) + 1)
        # y_test = np.asarray(y_test, dtype=np.int64)
        #
        # print('X_train.shape',X_train.shape)
        # print('y_train.shape', y_train.shape)
        # print('X_val.shape', X_val.shape)
        # print('y_val.shape', y_val.shape)
        # print('X_test.shape', X_test.shape)
        # print('y_test.shape', y_test.shape)
        # print('X %s   y %s' % (X_test.dtype, y_test.dtype))

        #,log_device_placement=True
        config_tf = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.InteractiveSession(config=config_tf)
        tl.ops.set_gpu_fraction(gpu_fraction=0.8)
        with tf.device('/cpu:0'):
            gnet = GoogleNetStructure(classes=y_train.shape[1],batch_size=200,imagewidth=X_train.shape[1],imageheight=X_train.shape[2],imagechannel=X_train.shape[3])
            gnet.model()
            if TrainAndSave == 1:
                gnet.fit(sess,True,X_train,y_train,gnet.x,gnet.y_,print_freq=5,n_epoch=500,x_val=None,y_val=None)
                #gnet.network.print_params()
                tl.files.exists_or_mkdir('model/')
                tl.files.save_npz(save_list=gnet.network.all_params, name="model/model5.npz", sess=sess)
            else:
                isRestore = False
                if tl.files.load_and_assign_npz(sess, name='model/model5.npz', network=gnet.network) == False:
                    saver = tf.train.Saver()
                    tl.layers.initialize_global_variables(sess)  # 变量的初始化
                    ckpt = tf.train.get_checkpoint_state('./saver/')
                    if ckpt and ckpt.model_checkpoint_path:
                        print("saver restore from %s" % (ckpt.model_checkpoint_path))
                        saver.restore(sess, ckpt.model_checkpoint_path)
                        isRestore = True
                    else:
                        print("model restore failed !")
                else:
                    isRestore = True
                if isRestore:
                    print("Starting predict ...")
                    res = gnet.predict(sess,X_test,y_test)
                    shape1 = res.shape[0]
                    shape2 = res.shape[1]
                    shape3 = res.shape[2]
                    #print("shape1 %d ,shape2 %d ,shape3 %d " %(shape1,shape2,shape3))
                    print("Summary data...")
                    global_num,right_num,error_num=0,0,0
                    for i in range(shape1):
                        for j in range(shape3):
                            global_num +=1
                            if res[i,0,j] == res[i,1,j]:
                                right_num += 1
                            else:
                                error_num +=1

                    print("global_num is %d |right_num is %d |error_num is %d " %(global_num,right_num,error_num))
                    print("                 |acc       is %f |loss      is %f " %(right_num/global_num,error_num/global_num))
                else:
                    print("exit")
        sess.close()
    except:
        import  sys
        data = sys.exc_info()
        print("系统错误:{}".format(data))

