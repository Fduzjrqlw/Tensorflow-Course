from __future__ import print_function
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data



def extract_samples(data) :
    '''
    考虑使用logistic regression回归,因此抽取出label为0和1的样本进行训练和预测.
    '''
    index_list = []
    for sample_index in range(data.shape[0]) :
        if (data[sample_index] == 0 or data[sample_index] == 1) :
            index_list.append(sample_index)
    return index_list

if __name__ == "__main__" :
    mnist =  input_data.read_data_sets('MNIST_data/' , reshape = True , one_hot = False)
    data = {}
    data['train/image'] = mnist.train.images
    data['train/label'] = mnist.train.labels
    data['test/image'] = mnist.test.images
    data['test/label'] = mnist.test.labels

    index_list = extract_samples(data['train/label'])
    data['train_used/image'] = data['train/image'][index_list]
    data['train_used/label'] = data['train/label'][index_list]

    index_list = extract_samples(data['test/label'])
    data['test_used/image'] = data['test/image'][index_list]
    data['test_used/label'] = data['test/label'][index_list]


    num_instances = data['train_used/image'].shape[0]
    num_features = data['train_used/image'].shape[1]

    image_place = tf.placeholder(tf.float32 , [None , num_features] , name = 'image')
    label_place = tf.placeholder(tf.int32 , [None] , name = 'label')
    label_one_hot = tf.one_hot(label_place , 2 , axis = -1)


    #注意logits的含义,是神经网络模型接的输出,作为未归一化的得分.
    #tf.reduce_mean和tf.reduce_sum的使用方式,axis的选择
    logits = tf.contrib.layers.fully_connected(inputs = image_place , num_outputs = 2)
    score = tf.nn.softmax(logits)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits , labels = label_one_hot))
    cross = tf.reduce_sum(-1 * tf.log(tf.nn.softmax(logits)) * label_one_hot , axis = -1)
    loss2 = tf.reduce_mean(cross)

    prediction_correct = tf.equal(tf.argmax(logits , 1) , tf.argmax(label_one_hot , 1))
    accuracy = tf.reduce_mean(tf.cast(prediction_correct , tf.float32))

    train_op = tf.train.GradientDescentOptimizer(learning_rate = 1e-3).minimize(loss)

    num_epoches = 10
    batch_size = 64

    with tf.Session() as sess :
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        print (sess.run(label_one_hot , feed_dict = {image_place : data['train_used/image'][ : 10] ,
                                                     label_place : data['train_used/label'][ : 10]}))
        print (sess.run(logits , feed_dict = {image_place : data['train_used/image'][ : 10] ,
                                                     label_place : data['train_used/label'][ : 10]}))
        print (sess.run(score , feed_dict = {image_place : data['train_used/image'][ : 10] ,
                                                     label_place : data['train_used/label'][ : 10]}))

        print (sess.run(cross , feed_dict = {image_place : data['train_used/image'][ : 10] ,
                                                     label_place : data['train_used/label'][ : 10]}))

        print (sess.run([loss , loss2] , feed_dict = {image_place : data['train_used/image'][ : 10] ,
                                                     label_place : data['train_used/label'][ : 10]}))

        for epoch in range(num_epoches) :
            total_batch = int(num_instances / batch_size)
            for batch in range(total_batch) :
                start_index = batch * batch_size
                end_index = (batch + 1) * batch_size

                train_batch_image = data['train_used/image'][start_index : end_index]
                train_batch_label = data['train_used/label'][start_index : end_index]

                batch_loss , _ = sess.run([loss , train_op] , feed_dict = {image_place : train_batch_image ,
                                                                           label_place : train_batch_label})
            print ("Epoch " + str(epoch + 1) + " , training_loss = " + "{: .5f}".format(batch_loss))

        test_accuray = 100 * sess.run(accuracy , feed_dict = {image_place : data['test_used/image'] ,
                                                        label_place : data['test_used/label']})

        print ("Final Test Accuracy is %% %.2f" % test_accuray)

    sess.close()