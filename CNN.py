import os
import tensorflow as tf
import numpy as np
import time
import inspect

slim = tf.contrib.slim

class CNN_net:
    def __init__(self, is_training=True):
        self.is_training = is_training
        self.image_size = 42
        self.class_size = 7
        self.learning_rate = 0.001
        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 1])
        self.logits = self.build(self.images, is_training=self.is_training)

        if self.is_training:
            self.labels = tf.placeholder(tf.float32, [None, self.class_size])
            # self.total_loss = slim.losses.softmax_cross_entropy(self.logits, self.labels)
            self.total_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels))
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss)

            # self.ema = tf.train.ExponentialMovingAverage(decay=0.9999)
            # self.averages_op = self.ema.apply(tf.trainable_variables())
            #
            # with tf.control_dependencies([self.optimizer1]):
            #     self.optimizer = tf.group(self.averages_op)

    def build(self, inputs, is_training):
        start_time = time.time()
        print("build cnn model started")

        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            conv1 = slim.conv2d(inputs, 32, [5, 5], padding='SAME', scope='conv1')
            pad1 = tf.pad(conv1, np.array([[0, 0], [0, 1], [0, 1], [0, 0]]), name='pad_1')
            pool1 = slim.max_pool2d(pad1, 3, 2, scope='pool1')
            conv2 = slim.conv2d(pool1, 32, [5, 5], padding='SAME')
            pad2 = tf.pad(conv2, np.array([[0, 0], [0, 1], [0, 1], [0, 0]]), name='pad_2')
            pool2 = slim.max_pool2d(pad2, 3, 2, scope='pool2')
            conv3 = slim.conv2d(pool2, 64, [5, 5], padding='SAME')
            pad3 = tf.pad(conv3, np.array([[0, 0], [0, 1], [0, 1], [0, 0]]), name='pad_3')
            pool3 = slim.max_pool2d(pad3, 3, 2, scope='pool3')

            # trans = tf.transpose(pool3, [0, 3, 1, 2], name='trans_1')
            flat = slim.flatten(pool3, scope='flat_1')
            full1 = slim.fully_connected(flat, 2048, scope='fc1')
            if is_training:
                full1 = slim.dropout(full1, 0.6, is_training=is_training, scope='dropout1')
            full2 = slim.fully_connected(full1, 1024, scope='fc2')
            if is_training:
                full2 = slim.dropout(full2, 0.6, is_training=is_training, scope='dropout2')
            softmax = slim.fully_connected(full2, self.class_size, activation_fn=None, scope='softmax')
            # print(conv1)
            # print(pool1)
            # print(conv2)
            # print(pool2)
            # print(conv3)
            # print(pool3)
            # print(full1)
            # print(full2)
            # print(softmax)
            print(("build model finished: %ds" % (time.time() - start_time)))
        return softmax