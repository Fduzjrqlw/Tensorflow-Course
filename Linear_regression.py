from __future__ import print_function
import pandas as pd
import tensorflow as tf
import numpy as np
import os


if __name__ == "__main__" :
    n = 50

    XX = np.arange(n)
    YY = 2.0 * XX + np.random.standard_normal(XX.shape)

    training_data = np.stack((XX , YY) , axis = -1)
    num_epochs = 100

    weights = tf.Variable(tf.zeros([1 , 1]) , name = 'weights')
    bias = tf.Variable(tf.zeros([1]) , name = 'bias')


    X = tf.placeholder(tf.float32 , [None , 1] , name = 'X')
    Y = tf.placeholder(tf.float32 , [None , 1] , name = 'Y')

    y_predict = tf.add(tf.matmul(X , weights) , bias)
    train_loss = tf.reduce_mean(tf.squared_difference(Y, y_predict)) * 0.5

    #不同的优化器要对应不同的学习率.sgd需选用1e-3的学习率,而Adam需选用1e-1
    train_op = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(train_loss)
    train_op = tf.train.AdamOptimizer(learning_rate=1e-1).minimize(train_loss)

    with tf.Session() as sess :
        sess.run(tf.global_variables_initializer())
        for epoch_num in range(num_epochs) :
            loss_ , _ = sess.run([train_loss , train_op] , feed_dict = {X : training_data[ : , 0].reshape(-1 , 1) ,
                                                                        Y : training_data[ : , 1].reshape(-1 , 1)})
            print ('epoch_num %d , loss = %f ' % (epoch_num , loss_))

    sess.close()