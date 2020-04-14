from __future__ import print_function
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import datasets
import os



if __name__ == "__main__" :
    iris = datasets.load_iris()

    #只用前2个feature
    X = iris.data[ : , ]
    Y = np.array([1 if label == 0 else -1 for label in iris.target]).reshape(-1 , 1)

    my_randoms = np.random.choice(X.shape[0] , X.shape[0] , replace = False)

    train_indices = my_randoms[ : int(0.7 * X.shape[0])]
    test_indices = my_randoms[int(0.7 * X.shape[0]) : ]

    X_train = X[train_indices]
    y_train = Y[train_indices]
    X_test = X[test_indices]
    y_test = Y[test_indices]


    X_feature = tf.placeholder(tf.float32 , [None , X.shape[1]] ,name = 'X_features')
    y_label = tf.placeholder(tf.float32 , [None , 1] , name = 'y_label' , )
    Weights = tf.Variable(tf.random_normal([X.shape[1] , 1] , mean = 0.0 , stddev = 1.0) , name = 'Weights')
    Bias = tf.Variable(tf.zeros([1 , 1]) , name = 'Bias')

    #注意写法 X * W + B
    logits = tf.add(tf.matmul(X_feature , Weights) , Bias)

    classification_loss = tf.reduce_mean(tf.maximum(0.0 , 1.0 - tf.multiply(y_label , logits)))
    normal_term = tf.divide(tf.reduce_sum(tf.matmul(tf.transpose(Weights) , Weights)) , 2.0)

    C_param = 0.1
    #选择合适的L2正则项系数
    Reg_param = 0.005

    total_loss = tf.add(tf.multiply(C_param , classification_loss) , tf.multiply(Reg_param , normal_term))
    init_op = tf.global_variables_initializer()

    train_op = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(total_loss)

    differ = tf.cast(tf.equal(tf.sign(logits) , y_label) , tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.sign(logits) , y_label) , tf.float32))

    with tf.Session() as sess :
        sess.run(init_op)
        for epoch in range(1000) :
            loss , acc , _ = sess.run([total_loss , accuracy , train_op] , feed_dict = {X_feature : X_train , y_label : y_train})
            if (epoch % 50 == 0) :
                print ("Epoch " + str(epoch + 1) + " : total_loss = " + "{: .5f}".format(loss) + " , accuracy = " + "{: .5f}".format(acc))

        final_acc = sess.run(accuracy , feed_dict = {X_feature : X_test , y_label : y_test})
        print ("accuracy = " + "{: .5f}".format(final_acc))
    sess.close()




