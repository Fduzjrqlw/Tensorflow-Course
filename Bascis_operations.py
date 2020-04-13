from __future__ import print_function
import tensorflow as tf
import pandas as pd
import os

if __name__ == "__main__" :

    a = tf.constant(5.0 , name = 'a')
    b = tf.constant(10.0 , name = 'b')
    x = tf.add(a , b , name = 'x')
    y = tf.multiply(a , b , name = 'y')
    z = tf.div(a , b , name = 'z')
    w = tf.add(a , -b , name = 'z')
    with tf.Session() as sess :
        print (sess.run([a , b , x , y , z , w]))
    sess.close()
