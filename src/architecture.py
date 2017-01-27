# importing stuff
import numpy as np
import tensorflow as tf
import initializer

# defining fuctions
def layer_definer(x):
	# layer 1
	W_conv1 = initializer.weight_variable([5,5,1,32])
	b_conv1 = initializer.bias_variable([32])
	
	x_image = tf.reshape(x, [-1, 28, 28, 1])
	
	h_conv1 = tf.nn.relu(initializer.conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = initializer.max_pool_2x2(h_conv1)
	
	# layer 2
	W_conv2 = initializer.weight_variable([5, 5, 32, 64])
	b_conv2 = initializer.bias_variable([64])

	h_conv2 = tf.nn.relu(initializer.conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = initializer.max_pool_2x2(h_conv2)

	# layer 3
	W_fc1 = initializer.weight_variable([7*7*64, 1024])
	b_fc1 = initializer.bias_variable([1024])
	
	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
	
	# layer 4
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	
	# layer 5
	W_fc2 = initializer.weight_variable([1024, 10])
	b_fc2 = initializer.bias_variable([10])
	
	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
	
	return y_conv, keep_prob
