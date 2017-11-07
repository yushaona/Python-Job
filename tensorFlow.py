import  tensorflow as tf
# with tf.device('/gpu:0'):
#     hello = tf.constant('Hello Tensorflow')
#
# va1 = tf.Variable(tf.random_normal([2,3]),name="weights")
# va2 = tf.Variable([1,2,3,4,5],name='height')
# init_op = tf.initialize_all_variables()
# sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
# sess.run(init_op)
# print(sess.run(hello))
# saver = tf.train.Saver()
# saver.save(sess,r'D:/1.ckpt')

vay = tf.Variable([1,1,1,1,1],name='height')
saver = tf.train.Saver()
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
saver.restore(sess,r'D:/1.ckpt')

print(sess.run(vay))




