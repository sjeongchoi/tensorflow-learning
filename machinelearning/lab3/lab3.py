import tensorflow as tf
x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

W = tf.Variable(tf.random_uniform([1], -10.0, 10.0))
#m = n_samples = len(X)
# W = tf.placeholder(tf.float32)

X = tf.placeholder(tf.float32);
Y = tf.placeholder(tf.float32);

# hypothesis = tf.mul(X, W)
hypothesis = W * X

#cost = tf.reduce_sum(tf.pow(hypothesis - Y, 2))/(m)
cost = tf.reduce_mean(tf.square(hypothesis - Y))

descent = W - tf.mul(0.1, tf.reduce_mean(tf.mul((tf.mul(W, X) - Y), X)))
update = W.assign(descent)

init = tf.initialize_all_variables()

#W_val = []
#cost_val = []

sess = tf.Session()
sess.run(init)

for step in xrange(20):
    sess.run(update, feed_dict={X:x_data, Y:y_data})
    print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W)
#    print i*0.1, sess.run(cost, feed_dict={W: i*0.1})
#    W_val.append(i*0.1)
#    cost_val.append(sess.run(cost, feed_dict={W : i*0.1}))

#plt.plot(W_val, cost_val, "ro")
#plt.ylabel('Cost')
#plt.xlabel('W')
#plt.show()