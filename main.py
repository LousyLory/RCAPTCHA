# necessary imports
import tensorflow as tf
import numpy as np
import src
from src.architecture import *

# reading the data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# start the session
sess = tf.InteractiveSession()

# variable initializer
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

y_conv, keep_prob = src.architecture.layer_definer(x)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

for i in range(2000):
	batch = mnist.train.next_batch(50)
	if i%100 == 0:
		train_accuracy = accuracy.eval(feed_dict={
				x:batch[0], y_: batch[1], keep_prob: 1.0})
		print("step %d, training accuracy %g"%(i, train_accuracy))
	train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

test_accuracy = 0.0
for i in range(2000):
        test_batch = mnist.test.next_batch(50)
        if i%100 == 0:
                test_accuracy += accuracy.eval(feed_dict={
                                x:test_batch[0], y_: test_batch[1], keep_prob: 1.0})
                print("step %d, cumulative test accuracy %g"%(i, test_accuracy/(i/100 + 1)))

#print("test accuracy %g"%accuracy.eval(feed_dict={
#    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
file_writer = tf.summary.FileWriter('logs/', sess.graph)
