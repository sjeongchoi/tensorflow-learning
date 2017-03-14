# coding=utf-8
import tensorflow as tf
import numpy as np


xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = np.transpose(xy[0:3])
y_data = np.transpose(xy[3:])

#tf Graph Input
X = tf.placeholder(tf.float32, [None, 3])
Y = tf.placeholder(tf.float32, [None, 3])

# set module weight
W = tf.Variable(tf.zeros([3, 3]))

# Hypothesis   -> w, x-> x, w
#다들 문제없으신 것 같지만, 혹시나 뒤에 하시는분들을 위해 적어보면,
#lecture slide 에는 hypothesis = tf.nn.softmax(tf.matmul(W, X)) 로 명시되어 있지만, 실제로 이와같이 돌리면 에러가 발생합니다. 교수님 실습 화면에 잠깐 나오지만 실제로 입력되어야 하는 값은  matmul(X, W) 입니다. 당연하지만 3X3 matrix 와 8X3 matrix 의 곱은 불가능하지만, 8X3 과 3X3 은 가능하기때문에 연산이 가능한 것 같습니다.﻿

hypothesis = tf.nn.softmax(tf.matmul(X, W))

learning_rate = 0.001

# cost entropy
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), reduction_indices=1))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Before starting, initialize the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    for step in xrange(2001):
        sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
        if step % 200 == 0:
            print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W)
    print "----------------------------"
    a=sess.run(hypothesis, feed_dict={X:[[1, 11, 7]]})
    print a, sess.run(tf.arg_max(a, 1))
    b=sess.run(hypothesis, feed_dict={X:[[1, 3, 4]]})
    print b, sess.run(tf.arg_max(b, 1))
    c=sess.run(hypothesis, feed_dict={X:[[1, 1, 0]]})
    print c, sess.run(tf.arg_maxa(c, 1))

    all=sess.run(hypothesis, feed_dict={X:[[1, 11, 7], [1, 3, 4], [1, 1, 0]]})
    print all, sess.run(tf.arg_max(all, 1))