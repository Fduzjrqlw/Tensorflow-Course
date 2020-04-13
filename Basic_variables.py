from __future__ import print_function
import tensorflow as tf
import pandas as pd
import os
from tensorflow.python.framework import ops


if __name__ == "__main__" :
    weights = tf.Variable(tf.random_normal([5 , 4] , mean = 0.0 , stddev = 1.0) , name = 'weights')
    bias = tf.Variable(tf.zeros([4]) , name = 'bias')

    all_variables_list = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)

    variables_list_custom = [weights]

    '''
    对于Variable类型的张量,要初始化.初始化有tf.variables_initializer和tf.global_variables_initializer两种方法,
    其中第一个方法需要指定张量列表.
    '''

    init_op = tf.variables_initializer(variables_list_custom)
    init_op1 = tf.global_variables_initializer()
    init_op2 = tf.variables_initializer(all_variables_list)

    weights_new = tf.Variable(initial_value = weights.initialized_value() , name = 'weights_new')
    init_op_new = tf.variables_initializer([weights_new])

    with tf.Session() as sess :
        sess.run(init_op)
        print(sess.run(weights))
        sess.run(init_op2)
        print (sess.run(weights))
        print (sess.run(bias))
        sess.run(init_op_new)
        print(sess.run(weights_new))
    sess.close()

