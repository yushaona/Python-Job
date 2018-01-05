import tensorflow as tf
import tensorlayer as tl

def GoogleNet(x,classes):
    with tf.variable_scope("googleNet"):
        W_init = tf.truncated_normal_initializer(stddev=5e-2)
        W_init2 = tf.truncated_normal_initializer(stddev=0.04)
        b_init2 = tf.constant_initializer(value=0.1)

        network = tl.layers.InputLayer(x,name='input_layer')
        conv1_7_7 = tl.layers.Conv2d(network, n_filter=64, filter_size=(7, 7), strides=(2, 2), act=tf.nn.relu,W_init=W_init,
                                     name='conv1_7_7_s2')
        pool1_3_3 = tl.layers.MaxPool2d(conv1_7_7, filter_size=(3, 3), strides=(2, 2),name='pool1_3_3')

        conv2_3_3_reduce = tl.layers.Conv2d(pool1_3_3, n_filter=64, strides=(1, 1), act=tf.nn.relu,W_init=W_init,
                                            name='conv2_3_3_reduce')
        conv2_3_3 = tl.layers.Conv2d(conv2_3_3_reduce, n_filter=192, filter_size=(3, 3), act=tf.nn.relu,W_init=W_init,
                                     name='conv2_3_3')

        pool2_3_3 = tl.layers.MaxPool2d(conv2_3_3, filter_size=(3, 3), strides=(2, 2), name='pool2_3_3_s2')


        inception_3a_1_1 = tl.layers.Conv2d(pool2_3_3, n_filter=64, filter_size=(1, 1), act=tf.nn.relu,W_init=W_init,
                                            name='inception_3a_1_1')
        inception_3a_3_3_reduce = tl.layers.Conv2d(pool2_3_3, n_filter=96, filter_size=(1, 1), act=tf.nn.relu,W_init=W_init,
                                                   name='inception_3a_3_3_reduce')
        inception_3a_3_3 = tl.layers.Conv2d(inception_3a_3_3_reduce, n_filter=128, filter_size=(3, 3), act=tf.nn.relu,W_init=W_init,
                                            name='inception_3a_3_3')
        inception_3a_5_5_reduce = tl.layers.Conv2d(pool2_3_3, n_filter=16, filter_size=(1, 1), act=tf.nn.relu,W_init=W_init,
                                                   name='inception_3a_5_5_reduce')
        inception_3a_5_5 = tl.layers.Conv2d(inception_3a_5_5_reduce, n_filter=32, filter_size=(3, 3), act=tf.nn.relu,W_init=W_init,
                                            name='inception_3a_5_5')
        inception_3a_pool = tl.layers.MaxPool2d(pool2_3_3, filter_size=(3, 3), strides=(1, 1), name='inception_3a_pool')
        inception_3a_pool_1_1 = tl.layers.Conv2d(inception_3a_pool, n_filter=32, filter_size=(1, 1), act=tf.nn.relu,W_init=W_init,
                                                 name='inception_3a_pool_1_1')
        inception_3a_output = tl.layers.ConcatLayer(
            [inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1], concat_dim=3,
            name="inception_3a_output")


        inception_3b_1_1 = tl.layers.Conv2d(inception_3a_output, n_filter=128, filter_size=(1, 1), act=tf.nn.relu,W_init=W_init,
                                            name='inception_3b_1_1')
        inception_3b_3_3_reduce = tl.layers.Conv2d(inception_3a_output, n_filter=128, filter_size=(1, 1),W_init=W_init,
                                                   act=tf.nn.relu, name='inception_3b_3_3_reduce')
        inception_3b_3_3 = tl.layers.Conv2d(inception_3b_3_3_reduce, n_filter=192, filter_size=(3, 3), act=tf.nn.relu,W_init=W_init,
                                            name='inception_3b_3_3')
        inception_3b_5_5_reduce = tl.layers.Conv2d(inception_3a_output, n_filter=32, filter_size=(1, 1), act=tf.nn.relu,W_init=W_init,
                                                   name='inception_3b_5_5_reduce')
        inception_3b_5_5 = tl.layers.Conv2d(inception_3b_5_5_reduce, n_filter=96, filter_size=(3, 3),W_init=W_init,
                                            name='inception_3b_5_5')
        inception_3b_pool = tl.layers.MaxPool2d(inception_3a_output, filter_size=(3, 3), strides=(1, 1),
                                                name='inception_3b_pool')
        inception_3b_pool_1_1 = tl.layers.Conv2d(inception_3b_pool, n_filter=64, filter_size=(1, 1), act=tf.nn.relu,W_init=W_init,
                                                 name='inception_3b_pool_1_1')
        inception_3b_output = tl.layers.ConcatLayer(
            [inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1],
            concat_dim=3, name="inception_3b_output")
        pool3_3_3 = tl.layers.MaxPool2d(inception_3b_output, filter_size=(3, 3), strides=(2, 2), name='pool3_3_3')

        inception_4a_1_1 = tl.layers.Conv2d(pool3_3_3, n_filter=192, filter_size=(1, 1), act=tf.nn.relu,W_init=W_init,
                                            name='inception_4a_1_1')

        inception_4a_3_3_reduce = tl.layers.Conv2d(pool3_3_3, n_filter=96, filter_size=(1, 1), act=tf.nn.relu,W_init=W_init,
                                                   name='inception_4a_3_3_reduce')
        inception_4a_3_3 = tl.layers.Conv2d(inception_4a_3_3_reduce, n_filter=208, filter_size=(3, 3), act=tf.nn.relu,W_init=W_init,
                                            name='inception_4a_3_3')

        inception_4a_5_5_reduce = tl.layers.Conv2d(pool3_3_3, n_filter=16, filter_size=(1, 1), act=tf.nn.relu,W_init=W_init,
                                                   name='inception_4a_5_5_reduce')
        inception_4a_5_5 = tl.layers.Conv2d(inception_4a_5_5_reduce, n_filter=48, filter_size=(3, 3), act=tf.nn.relu,W_init=W_init,
                                            name='inception_4a_5_5')

        inception_4a_pool = tl.layers.MaxPool2d(pool3_3_3, filter_size=(3, 3), strides=(1, 1), name='inception_4a_pool')
        inception_4a_pool_1_1 = tl.layers.Conv2d(inception_4a_pool, n_filter=64, filter_size=(1, 1), act=tf.nn.relu,W_init=W_init,
                                                 name='inception_4a_pool_1_1')
        # merge the inception_4a_*
        inception_4a_output = tl.layers.ConcatLayer(
            [inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1], concat_dim=3,
            name='inception_4a_output')

        #####inception(4b)######
        inception_4b_1_1 = tl.layers.Conv2d(inception_4a_output, n_filter=160, filter_size=(1, 1), act=tf.nn.relu,W_init=W_init,
                                            name='inception_4b_1_1')
        inception_4b_3_3_reduce = tl.layers.Conv2d(inception_4a_output, n_filter=112, filter_size=(1, 1),W_init=W_init,
                                                   act=tf.nn.relu, name='inception_4b_3_3_reduce')
        inception_4b_3_3 = tl.layers.Conv2d(inception_4b_3_3_reduce, n_filter=224, filter_size=(3, 3), act=tf.nn.relu,W_init=W_init,
                                            name='inception_4b_3_3')

        inception_4b_5_5_reduce = tl.layers.Conv2d(inception_4a_output, n_filter=24, filter_size=(1, 1), act=tf.nn.relu,W_init=W_init,
                                                   name='inception_4b_5_5_reduce')
        inception_4b_5_5 = tl.layers.Conv2d(inception_4b_5_5_reduce, n_filter=64, filter_size=(3, 3), act=tf.nn.relu,W_init=W_init,
                                            name='inception_4b_5_5')

        inception_4b_pool = tl.layers.MaxPool2d(inception_4a_output, filter_size=(3, 3), strides=(1, 1),
                                                name='inception_4b_pool')
        inception_4b_pool_1_1 = tl.layers.Conv2d(inception_4b_pool, n_filter=64, filter_size=(1, 1), act=tf.nn.relu,W_init=W_init,
                                                 name='inception_4b_pool_1_1')
        # merge the inception_4b_*
        inception_4b_output = tl.layers.ConcatLayer(
            [inception_4b_1_1, inception_4b_3_3, inception_4b_5_5, inception_4b_pool_1_1], concat_dim=3,
            name='inception_4b_output')

        #####inception(4c)######
        inception_4c_1_1 = tl.layers.Conv2d(inception_4b_output, n_filter=128, filter_size=(1, 1), act=tf.nn.relu,W_init=W_init,
                                            name='inception_4c_1_1')

        inception_4c_3_3_reduce = tl.layers.Conv2d(inception_4b_output, n_filter=128, filter_size=(1, 1),W_init=W_init,
                                                   act=tf.nn.relu, name='inception_4c_3_3_reduce')
        inception_4c_3_3 = tl.layers.Conv2d(inception_4c_3_3_reduce, n_filter=256, filter_size=(3, 3), act=tf.nn.relu,W_init=W_init,
                                            name='inception_4c_3_3')

        inception_4c_5_5_reduce = tl.layers.Conv2d(inception_4b_output, n_filter=24, filter_size=(1, 1), act=tf.nn.relu,W_init=W_init,
                                                   name='inception_4c_5_5_reduce')
        inception_4c_5_5 = tl.layers.Conv2d(inception_4c_5_5_reduce, n_filter=64, filter_size=(3, 3), act=tf.nn.relu,W_init=W_init,
                                            name='inception_4c_5_5')

        inception_4c_pool = tl.layers.MaxPool2d(inception_4b_output, filter_size=(3, 3), strides=(1, 1),
                                                name='inception_4c_pool')
        inception_4c_pool_1_1 = tl.layers.Conv2d(inception_4c_pool, n_filter=64, filter_size=(1, 1), act=tf.nn.relu,W_init=W_init,
                                                 name='inception_4c_pool_1_1')
        # merge the inception_4c_*
        inception_4c_output = tl.layers.ConcatLayer(
            [inception_4c_1_1, inception_4c_3_3, inception_4c_5_5, inception_4c_pool_1_1], concat_dim=3,
            name='inception_4c_output')

        #####inception(4d)######
        inception_4d_1_1 = tl.layers.Conv2d(inception_4c_output, n_filter=112, filter_size=(1, 1), act=tf.nn.relu,W_init=W_init,
                                            name='inception_4d_1_1')

        inception_4d_3_3_reduce = tl.layers.Conv2d(inception_4c_output, n_filter=144, filter_size=(1, 1),W_init=W_init,
                                                   act=tf.nn.relu, name='inception_4d_3_3_reduce')
        inception_4d_3_3 = tl.layers.Conv2d(inception_4d_3_3_reduce, n_filter=288, filter_size=(3, 3), act=tf.nn.relu,W_init=W_init,
                                            name='inception_4d_3_3')

        inception_4d_5_5_reduce = tl.layers.Conv2d(inception_4c_output, n_filter=32, filter_size=(1, 1), act=tf.nn.relu,W_init=W_init,
                                                   name='inception_4d_5_5_reduce')
        inception_4d_5_5 = tl.layers.Conv2d(inception_4d_5_5_reduce, n_filter=64, filter_size=(3, 3), act=tf.nn.relu,W_init=W_init,
                                            name='inception_4d_5_5')

        inception_4d_pool = tl.layers.MaxPool2d(inception_4c_output, filter_size=(3, 3), strides=(1, 1),
                                                name='inception_4d_pool')
        inception_4d_pool_1_1 = tl.layers.Conv2d(inception_4d_pool, n_filter=64, filter_size=(1, 1), act=tf.nn.relu,W_init=W_init,
                                                 name='inception_4d_pool_1_1')

        # merge the inception_4d_*
        inception_4d_output = tl.layers.ConcatLayer(
            [inception_4d_1_1, inception_4d_3_3, inception_4d_5_5, inception_4d_pool_1_1], concat_dim=3,
            name='inception_4d_output')

        #####inception(4e)######
        inception_4e_1_1 = tl.layers.Conv2d(inception_4d_output, n_filter=256, filter_size=(1, 1), act=tf.nn.relu,W_init=W_init,
                                            name='inception_4e_1_1')

        inception_4e_3_3_reduce = tl.layers.Conv2d(inception_4d_output, n_filter=160, filter_size=(1, 1),W_init=W_init,
                                                   act=tf.nn.relu, name='inception_4e_3_3_reduce')
        inception_4e_3_3 = tl.layers.Conv2d(inception_4e_3_3_reduce, n_filter=320, filter_size=(3, 3), act=tf.nn.relu,W_init=W_init,
                                            name='inception_4e_3_3')

        inception_4e_5_5_reduce = tl.layers.Conv2d(inception_4d_output, n_filter=32, filter_size=(1, 1), act=tf.nn.relu,W_init=W_init,
                                                   name='inception_4e_5_5_reduce')
        inception_4e_5_5 = tl.layers.Conv2d(inception_4e_5_5_reduce, n_filter=128, filter_size=(3, 3), act=tf.nn.relu,W_init=W_init,
                                            name='inception_4e_5_5')

        inception_4e_pool = tl.layers.MaxPool2d(inception_4d_output, filter_size=(3, 3), strides=(1, 1),
                                                name='inception_4e_pool')
        inception_4e_pool_1_1 = tl.layers.Conv2d(inception_4e_pool, n_filter=128, filter_size=(1, 1), act=tf.nn.relu,W_init=W_init,
                                                 name='inception_4e_pool_1_1')

        # merge the inception_4e_*
        inception_4e_output = tl.layers.ConcatLayer(
            [inception_4e_1_1, inception_4e_3_3, inception_4e_5_5, inception_4e_pool_1_1], concat_dim=3,
            name='inception_4e_output')
        pool4_3_3 = tl.layers.MaxPool2d(inception_4e_output, filter_size=(2, 2), strides=(2, 2), name='pool4_3_3')

        #####inception(5a)######
        inception_5a_1_1 = tl.layers.Conv2d(pool4_3_3, n_filter=256, filter_size=(1, 1), act=tf.nn.relu,W_init=W_init,
                                            name='inception_5a_1_1')

        inception_5a_3_3_reduce = tl.layers.Conv2d(pool4_3_3, n_filter=160, filter_size=(1, 1), act=tf.nn.relu,W_init=W_init,
                                                   name='inception_5a_3_3_reduce')
        inception_5a_3_3 = tl.layers.Conv2d(inception_5a_3_3_reduce, n_filter=320, filter_size=(3, 3), act=tf.nn.relu,W_init=W_init,
                                            name='inception_5a_3_3')

        inception_5a_5_5_reduce = tl.layers.Conv2d(pool4_3_3, n_filter=32, filter_size=(1, 1), act=tf.nn.relu,W_init=W_init,
                                                   name='inception_5a_5_5_reduce')
        inception_5a_5_5 = tl.layers.Conv2d(inception_5a_5_5_reduce, n_filter=128, filter_size=(3, 3), act=tf.nn.relu,W_init=W_init,
                                            name='inception_5a_5_5')

        inception_5a_pool = tl.layers.MaxPool2d(pool4_3_3, filter_size=(3, 3), strides=(1, 1), name='inception_5a_pool')
        inception_5a_pool_1_1 = tl.layers.Conv2d(inception_5a_pool, n_filter=128, filter_size=(1, 1), act=tf.nn.relu,W_init=W_init,
                                                 name='inception_5a_pool_1_1')

        # merge the inception_5a_*
        inception_5a_output = tl.layers.ConcatLayer(
            [inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1],
            concat_dim=3,
            name='inception_5a_output')

        #####inception(5b)######
        inception_5b_1_1 = tl.layers.Conv2d(inception_5a_output, n_filter=384, filter_size=(1, 1), act=tf.nn.relu,W_init=W_init,
                                            name='inception_5b_1_1')

        inception_5b_3_3_reduce = tl.layers.Conv2d(inception_5a_output, n_filter=192, filter_size=(1, 1),
                                                   act=tf.nn.relu,W_init=W_init, name='inception_5b_3_3_reduce')
        inception_5b_3_3 = tl.layers.Conv2d(inception_5b_3_3_reduce, n_filter=384, filter_size=(3, 3), act=tf.nn.relu,W_init=W_init,
                                            name='inception_5b_3_3')

        inception_5b_5_5_reduce = tl.layers.Conv2d(inception_5a_output, n_filter=48, filter_size=(1, 1), act=tf.nn.relu,W_init=W_init,
                                                   name='inception_5b_5_5_reduce')
        inception_5b_5_5 = tl.layers.Conv2d(inception_5b_5_5_reduce, n_filter=128, filter_size=(5, 5), act=tf.nn.relu,W_init=W_init,
                                            name='inception_5b_5_5')

        inception_5b_pool = tl.layers.MaxPool2d(inception_5a_output, filter_size=(3, 3), strides=(1, 1),
                                                name='inception_5b_pool')
        inception_5b_pool_1_1 = tl.layers.Conv2d(inception_5b_pool, n_filter=128, filter_size=(1, 1), act=tf.nn.relu,W_init=W_init,
                                                 name='inception_5b_pool_1_1')
        # merge the inception_5b_*
        inception_5b_output = tl.layers.ConcatLayer(
            [inception_5b_1_1, inception_5b_3_3, inception_5b_5_5, inception_5b_pool_1_1], concat_dim=3,
            name='inception_5b_output')

        pool5_7_7 = tl.layers.MeanPool2d(inception_5b_output, filter_size=(7, 7), strides=(1, 1), name='AvgPool2D')

        net = tl.layers.FlattenLayer(pool5_7_7, name='flatten_layer')
        #net = tl.layers.DropoutLayer(net, keep=0.8, name='dropout_layer_1')
        net = tl.layers.DenseLayer(net, n_units=1024, act=tf.nn.relu, W_init=W_init2, b_init=b_init2,
                                   name='FullyConnected_111')
        net = tl.layers.DenseLayer(net, n_units=256, act=tf.nn.relu,W_init=W_init2, b_init=b_init2, name='FullyConnected_1')
        #net = tl.layers.DropoutLayer(net, keep=0.5, name='dropout_layer_2')
        net = tl.layers.DenseLayer(net, n_units=classes,
                                       act=tf.identity,W_init=tf.truncated_normal_initializer(stddev=1/256.0),
                                       name='FullyConnected_2')
        return net

def Model(x,classes):
    with tf.variable_scope("googleNet_V"):
        W_init = tf.truncated_normal_initializer(stddev=5e-2)
        W_init2 = tf.truncated_normal_initializer(stddev=0.04)
        b_init2 = tf.constant_initializer(value=0.1)

        network = tl.layers.InputLayer(x, name='input_layer')
        conv1_7_7 = tl.layers.Conv2d(network,n_filter=64,filter_size=(7,7),strides=(2,2),act=tf.nn.relu,name='conv1_7_7_s2',W_init=W_init)
        pool1_3_3 = tl.layers.MaxPool2d(conv1_7_7,filter_size=(3,3),strides=(2,2))
        pool1_3_3 = tl.layers.LocalResponseNormLayer(pool1_3_3,depth_radius=5, bias=1.0,alpha=0.0001, beta=0.75,name='lrn_1')
        conv2_3_3_reduce = tl.layers.Conv2d(pool1_3_3,n_filter=64,strides=(1,1),act=tf.nn.relu,name='conv2_3_3_reduce',W_init=W_init)
        conv2_3_3 = tl.layers.Conv2d(conv2_3_3_reduce, n_filter=192, filter_size=(3,3), act=tf.nn.relu, name='conv2_3_3',W_init=W_init)
        conv2_3_3 = tl.layers.LocalResponseNormLayer(conv2_3_3, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75,name='lrn_2')
        pool2_3_3 = tl.layers.MaxPool2d(conv2_3_3,filter_size=(3,3),strides=(2,2), name='pool2_3_3_s2')

        ####inception(3a)####
        inception_3a_1_1 = tl.layers.Conv2d(pool2_3_3,n_filter=64,filter_size=(1,1),act=tf.nn.relu, name='inception_3a_1_1',W_init=W_init)

        inception_3a_3_3_reduce = tl.layers.Conv2d(pool2_3_3,n_filter= 96,filter_size=(1,1), act=tf.nn.relu, name='inception_3a_3_3_reduce',W_init=W_init)
        inception_3a_3_3 = tl.layers.Conv2d(inception_3a_3_3_reduce,n_filter= 128, filter_size=(3,3), act=tf.nn.relu, name='inception_3a_3_3',W_init=W_init)

        inception_3a_5_5_reduce = tl.layers.Conv2d(pool2_3_3,n_filter= 16, filter_size=(1,1), act=tf.nn.relu, name='inception_3a_5_5_reduce',W_init=W_init)
        inception_3a_5_5 = tl.layers.Conv2d(inception_3a_5_5_reduce,n_filter= 32, filter_size=(5,5), act=tf.nn.relu, name='inception_3a_5_5',W_init=W_init)

        inception_3a_pool = tl.layers.MaxPool2d(pool2_3_3,filter_size=(3,3), strides=(1,1),name='inception_3a_pool' )
        inception_3a_pool_1_1 = tl.layers.Conv2d(inception_3a_pool, n_filter=32, filter_size=(1,1), act=tf.nn.relu,name='inception_3a_pool_1_1',W_init=W_init)
        # merge the inception_3a_*
        inception_3a_output = tl.layers.ConcatLayer([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1],concat_dim=3,name="inception_3a_output")

        ####inception(3b)####
        inception_3b_1_1 = tl.layers.Conv2d(inception_3a_output, n_filter=128, filter_size=(1, 1), act=tf.nn.relu, name='inception_3b_1_1',W_init=W_init)

        inception_3b_3_3_reduce = tl.layers.Conv2d(inception_3a_output, n_filter=128, filter_size=(1, 1), act=tf.nn.relu, name='inception_3b_3_3_reduce',W_init=W_init)
        inception_3b_3_3 = tl.layers.Conv2d(inception_3b_3_3_reduce, n_filter=192, filter_size=(3,3), act=tf.nn.relu,name='inception_3b_3_3',W_init=W_init)

        inception_3b_5_5_reduce = tl.layers.Conv2d(inception_3a_output, n_filter=32, filter_size=(1,1), act=tf.nn.relu,name='inception_3b_5_5_reduce',W_init=W_init)
        inception_3b_5_5 = tl.layers.Conv2d(inception_3b_5_5_reduce,n_filter= 96, filter_size=(5,5), name='inception_3b_5_5',W_init=W_init)

        inception_3b_pool = tl.layers.MaxPool2d(inception_3a_output, filter_size=(3,3), strides=(1,1), name='inception_3b_pool')
        inception_3b_pool_1_1 = tl.layers.Conv2d(inception_3b_pool, n_filter=64, filter_size=(1,1), act=tf.nn.relu,name='inception_3b_pool_1_1',W_init=W_init)
        # merge the inception_3b_*
        inception_3b_output = tl.layers.ConcatLayer([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1],
                                                    concat_dim=3, name="inception_3b_output")
        # pool the inception_3b_*
        pool3_3_3 = tl.layers.MaxPool2d(inception_3b_output,filter_size=(3,3),strides=(2,2),name='pool3_3_3')


        #####inception(4a)######
        inception_4a_1_1 = tl.layers.Conv2d(pool3_3_3,n_filter=192,filter_size=(1,1),act=tf.nn.relu,name='inception_4a_1_1',W_init=W_init)

        inception_4a_3_3_reduce = tl.layers.Conv2d(pool3_3_3,n_filter=96,filter_size=(1,1),act=tf.nn.relu,name='inception_4a_3_3_reduce',W_init=W_init)
        inception_4a_3_3 = tl.layers.Conv2d(inception_4a_3_3_reduce, n_filter=208, filter_size=(3,3), act=tf.nn.relu,name='inception_4a_3_3',W_init=W_init)

        inception_4a_5_5_reduce = tl.layers.Conv2d(pool3_3_3,n_filter= 16, filter_size=(1,1), act=tf.nn.relu,name='inception_4a_5_5_reduce',W_init=W_init)
        inception_4a_5_5 = tl.layers.Conv2d(inception_4a_5_5_reduce,n_filter= 48, filter_size=(5,5), act=tf.nn.relu,name='inception_4a_5_5',W_init=W_init)

        inception_4a_pool = tl.layers.MaxPool2d(pool3_3_3, filter_size=(3,3), strides=(1,1), name='inception_4a_pool')
        inception_4a_pool_1_1 = tl.layers.Conv2d(inception_4a_pool,n_filter= 64, filter_size=(1,1),  act=tf.nn.relu,name='inception_4a_pool_1_1',W_init=W_init)

        # merge the inception_4a_*
        inception_4a_output = tl.layers.ConcatLayer([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1],concat_dim=3,name='inception_4a_output')

        #####inception(4b)######
        inception_4b_1_1 = tl.layers.Conv2d(inception_4a_output,n_filter= 160, filter_size=(1,1), act=tf.nn.relu, name='inception_4b_1_1',W_init=W_init)
        inception_4b_3_3_reduce = tl.layers.Conv2d(inception_4a_output, n_filter=112, filter_size=(1,1), act=tf.nn.relu,name='inception_4b_3_3_reduce',W_init=W_init)
        inception_4b_3_3 = tl.layers.Conv2d(inception_4b_3_3_reduce, n_filter=224, filter_size=(3,3), act=tf.nn.relu,name='inception_4b_3_3',W_init=W_init)

        inception_4b_5_5_reduce = tl.layers.Conv2d(inception_4a_output,n_filter= 24, filter_size=(1,1),act=tf.nn.relu,name='inception_4b_5_5_reduce',W_init=W_init)
        inception_4b_5_5 = tl.layers.Conv2d(inception_4b_5_5_reduce, n_filter=64, filter_size=(5,5), act=tf.nn.relu,name='inception_4b_5_5',W_init=W_init)

        inception_4b_pool = tl.layers.MaxPool2d(inception_4a_output, filter_size=(3,3), strides=(1,1), name='inception_4b_pool')
        inception_4b_pool_1_1 = tl.layers.Conv2d(inception_4b_pool, n_filter=64, filter_size=(1,1), act=tf.nn.relu,name='inception_4b_pool_1_1',W_init=W_init)
        # merge the inception_4b_*
        inception_4b_output = tl.layers.ConcatLayer([inception_4b_1_1, inception_4b_3_3, inception_4b_5_5, inception_4b_pool_1_1],concat_dim=3, name='inception_4b_output')

        #####inception(4c)######
        inception_4c_1_1 = tl.layers.Conv2d(inception_4b_output,n_filter= 128, filter_size=(1,1), act=tf.nn.relu, name='inception_4c_1_1',W_init=W_init)

        inception_4c_3_3_reduce = tl.layers.Conv2d(inception_4b_output, n_filter=128, filter_size=(1,1), act=tf.nn.relu,name='inception_4c_3_3_reduce',W_init=W_init)
        inception_4c_3_3 = tl.layers.Conv2d(inception_4c_3_3_reduce,n_filter= 256, filter_size=(3,3), act=tf.nn.relu,name='inception_4c_3_3',W_init=W_init)

        inception_4c_5_5_reduce = tl.layers.Conv2d(inception_4b_output,n_filter= 24, filter_size=(1,1), act=tf.nn.relu,name='inception_4c_5_5_reduce',W_init=W_init)
        inception_4c_5_5 = tl.layers.Conv2d(inception_4c_5_5_reduce,n_filter= 64, filter_size=(5,5), act=tf.nn.relu,name='inception_4c_5_5',W_init=W_init)

        inception_4c_pool = tl.layers.MaxPool2d(inception_4b_output,filter_size=(3,3), strides=(1,1),name='inception_4c_pool')
        inception_4c_pool_1_1 = tl.layers.Conv2d(inception_4c_pool,n_filter= 64, filter_size=(1,1), act=tf.nn.relu,name='inception_4c_pool_1_1',W_init=W_init)
        # merge the inception_4c_*
        inception_4c_output = tl.layers.ConcatLayer(
            [inception_4c_1_1, inception_4c_3_3, inception_4c_5_5, inception_4c_pool_1_1], concat_dim=3,
            name='inception_4c_output')

        #####inception(4d)######
        inception_4d_1_1 = tl.layers.Conv2d(inception_4c_output, n_filter=112, filter_size=(1,1), act=tf.nn.relu, name='inception_4d_1_1',W_init=W_init)

        inception_4d_3_3_reduce = tl.layers.Conv2d(inception_4c_output,n_filter= 144, filter_size=(1,1), act=tf.nn.relu,name='inception_4d_3_3_reduce',W_init=W_init)
        inception_4d_3_3 = tl.layers.Conv2d(inception_4d_3_3_reduce,n_filter= 288, filter_size=(3,3),  act=tf.nn.relu, name='inception_4d_3_3',W_init=W_init)

        inception_4d_5_5_reduce = tl.layers.Conv2d(inception_4c_output,n_filter= 32, filter_size=(1,1), act=tf.nn.relu,name='inception_4d_5_5_reduce',W_init=W_init)
        inception_4d_5_5 = tl.layers.Conv2d(inception_4d_5_5_reduce, n_filter=64, filter_size=(5,5),act=tf.nn.relu,name='inception_4d_5_5',W_init=W_init)


        inception_4d_pool = tl.layers.MaxPool2d(inception_4c_output,filter_size=(3,3), strides=(1,1), name='inception_4d_pool')
        inception_4d_pool_1_1 = tl.layers.Conv2d(inception_4d_pool, n_filter=64, filter_size=(1,1), act=tf.nn.relu,name='inception_4d_pool_1_1',W_init=W_init)

        # merge the inception_4d_*
        inception_4d_output = tl.layers.ConcatLayer(
            [inception_4d_1_1, inception_4d_3_3, inception_4d_5_5, inception_4d_pool_1_1], concat_dim=3,
            name='inception_4d_output')

        #####inception(4e)######
        inception_4e_1_1 = tl.layers.Conv2d(inception_4d_output,n_filter= 256, filter_size=(1,1), act=tf.nn.relu, name='inception_4e_1_1',W_init=W_init)

        inception_4e_3_3_reduce = tl.layers.Conv2d(inception_4d_output,n_filter= 160, filter_size=(1,1), act=tf.nn.relu,name='inception_4e_3_3_reduce',W_init=W_init)
        inception_4e_3_3 = tl.layers.Conv2d(inception_4e_3_3_reduce,n_filter= 320, filter_size=(3,3), act=tf.nn.relu,name='inception_4e_3_3',W_init=W_init)

        inception_4e_5_5_reduce = tl.layers.Conv2d(inception_4d_output, n_filter=32, filter_size=(1,1), act=tf.nn.relu,name='inception_4e_5_5_reduce',W_init=W_init)
        inception_4e_5_5 = tl.layers.Conv2d(inception_4e_5_5_reduce, n_filter=128, filter_size=(5,5), act=tf.nn.relu,name='inception_4e_5_5',W_init=W_init)


        inception_4e_pool = tl.layers.MaxPool2d(inception_4d_output,filter_size=(3,3),strides=(1,1), name='inception_4e_pool')
        inception_4e_pool_1_1 = tl.layers.Conv2d(inception_4e_pool,n_filter= 128, filter_size=(1,1), act=tf.nn.relu,name='inception_4e_pool_1_1',W_init=W_init)

        # merge the inception_4e_*
        inception_4e_output = tl.layers.ConcatLayer(
            [inception_4e_1_1, inception_4e_3_3, inception_4e_5_5, inception_4e_pool_1_1], concat_dim=3,
            name='inception_4e_output')
        pool4_3_3 = tl.layers.MaxPool2d(inception_4e_output, filter_size=(3,3), strides=(2,2), name='pool4_3_3')


        #####inception(5a)######
        inception_5a_1_1 = tl.layers.Conv2d(pool4_3_3,n_filter= 256, filter_size=(1,1), act=tf.nn.relu, name='inception_5a_1_1',W_init=W_init)

        inception_5a_3_3_reduce = tl.layers.Conv2d(pool4_3_3, n_filter=160, filter_size=(1,1),act=tf.nn.relu,name='inception_5a_3_3_reduce',W_init=W_init)
        inception_5a_3_3 = tl.layers.Conv2d(inception_5a_3_3_reduce, n_filter=320, filter_size=(3,3),act=tf.nn.relu,name='inception_5a_3_3',W_init=W_init)

        inception_5a_5_5_reduce = tl.layers.Conv2d(pool4_3_3, n_filter=32, filter_size=(1,1), act=tf.nn.relu,name='inception_5a_5_5_reduce',W_init=W_init)
        inception_5a_5_5 = tl.layers.Conv2d(inception_5a_5_5_reduce, n_filter=128, filter_size=(5,5), act=tf.nn.relu,name='inception_5a_5_5',W_init=W_init)

        inception_5a_pool = tl.layers.MaxPool2d(pool4_3_3, filter_size=(3,3), strides=(1,1), name='inception_5a_pool')
        inception_5a_pool_1_1 = tl.layers.Conv2d(inception_5a_pool, n_filter=128, filter_size=(1,1),act=tf.nn.relu,name='inception_5a_pool_1_1',W_init=W_init)

        # merge the inception_5a_*
        inception_5a_output = tl.layers.ConcatLayer(
            [inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1],
                                                    concat_dim=3,
                                                    name='inception_5a_output')

        #####inception(5b)######
        inception_5b_1_1 = tl.layers.Conv2d(inception_5a_output,n_filter= 384, filter_size=(1,1), act=tf.nn.relu, name='inception_5b_1_1',W_init=W_init)

        inception_5b_3_3_reduce = tl.layers.Conv2d(inception_5a_output,n_filter= 192, filter_size=(1,1), act=tf.nn.relu,name='inception_5b_3_3_reduce',W_init=W_init)
        inception_5b_3_3 = tl.layers.Conv2d(inception_5b_3_3_reduce,n_filter= 384, filter_size=(3,3),act=tf.nn.relu, name='inception_5b_3_3',W_init=W_init)

        inception_5b_5_5_reduce = tl.layers.Conv2d(inception_5a_output, n_filter=48, filter_size=(1,1), act=tf.nn.relu,name='inception_5b_5_5_reduce',W_init=W_init)
        inception_5b_5_5 =tl.layers.Conv2d(inception_5b_5_5_reduce, n_filter=128, filter_size=(5,5), act=tf.nn.relu, name='inception_5b_5_5',W_init=W_init)

        inception_5b_pool = tl.layers.MaxPool2d(inception_5a_output, filter_size=(3,3), strides=(1,1), name='inception_5b_pool')
        inception_5b_pool_1_1 = tl.layers.Conv2d(inception_5b_pool, n_filter=128, filter_size=(1,1),act=tf.nn.relu,name='inception_5b_pool_1_1',W_init=W_init)
        # merge the inception_5b_*
        inception_5b_output = tl.layers.ConcatLayer([inception_5b_1_1, inception_5b_3_3, inception_5b_5_5, inception_5b_pool_1_1], concat_dim=3,name='inception_5b_output')

        pool5_7_7 = tl.layers.MeanPool2d(inception_5b_output,filter_size=(7,7),strides=(1,1),name='AvgPool2D')

        pool5_7_7 = tl.layers.DropoutLayer(pool5_7_7, keep=0.8, name='dropout_layer_1')

        pool5_7_7 = tl.layers.FlattenLayer(pool5_7_7, name='flatten_layer')

        net = tl.layers.DenseLayer(pool5_7_7, n_units=1024, act=tf.nn.relu,
                                   name='FullyConnected_1', W_init=W_init2, b_init=b_init2)
        net = tl.layers.DenseLayer(net, n_units=256, act=tf.nn.relu,
                                   name='FullyConnected_2', W_init=W_init2, b_init=b_init2)
        net = tl.layers.DenseLayer(net, n_units=classes,
                                   act=tf.identity, W_init=tf.truncated_normal_initializer(stddev=1 / 256.0),
                                   name='FullyConnected_3')
        return  net




