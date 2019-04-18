""" Starter code for simple linear regression example using placeholders
Created by Chip Huyen (huyenn@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import utils

DATA_FILE = '/home/colinlu/projects/cs224/lecture_code/birth_life_2010.txt'

# Step 1: read in data from the .txt file
data, n_samples = utils.read_birth_life_data(DATA_FILE)

# Step 2: create placeholders for X (birth rate) and Y (life expectancy)
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# Step 3: create weight and bias, initialized to 0.0
w = tf.get_variable('weight', initializer=tf.constant(0.0))
b = tf.get_variable('bias', initializer=tf.constant(0.0))

# Step 4: build model to predict Y
Y_predicted = w * X + b

# Step 5: use the square error as the loss function
loss = tf.square(Y - Y_predicted, name='loss')

# Step 6: using gradient descent with learning rate of 0.001 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

start = time.time()
writer = tf.summary.FileWriter('./graphs/linear_reg', tf.get_default_graph())
print(X, Y, w, b)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # Step 8: train the model for 100 epochs
    for i in range(100):
        total_loss = 0
        for x, y in data:
            _, loss = sess.run([optimizer, loss], feed_dict={X: x, Y:y})
            total_loss += loss
        print('Epoch {0}: {1}'.format(i, total_loss/n_samples))

    writer.close()
    
    # Step 9: output the values of w and b
    w_out, b_out = sess.run([w, b])

print('Took: %f seconds' %(time.time() - start))

# uncomment the following lines to see the plot 
# plt.plot(data[:,0], data[:,1], 'bo', label='Real data')
# plt.plot(data[:,0], data[:,0] * w_out + b_out, 'r', label='Predicted data')
# plt.legend()
# plt.show()
