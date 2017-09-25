"""
Simple tester for the vgg19_trainable
"""

import tensorflow as tf
import datetime
from utils.timer import Timer
from utils.fer2013 import Fer2013
import ImageNet
import numpy as np

with tf.Session() as sess:
    cnn = ImageNet.CNN_net(False)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=1)

    model_file = tf.train.latest_checkpoint('ckpt/')
    saver.restore(sess, model_file)

    fer2013_test = Fer2013('test')

    for step in range(10):

        images_test, labels_test = fer2013_test.get_test()
        feed_dict_test = {cnn.images: images_test}
        logits_test = sess.run(cnn.logits, feed_dict=feed_dict_test)
        logits_final = np.zeros((fer2013_test.batch_size, fer2013_test.classesNum), dtype='float')
        for i in range(fer2013_test.batch_size):
            # logits_all = logits_test[i*8+0]+logits_test[i*8+1]+logits_test[i*8+2]+logits_test[i*8+3]+ \
            #              logits_test[i*8+4]+logits_test[i*8+5]+logits_test[i*8+6]+logits_test[i*8+7]
            logits_all = logits_test[i * 4 + 0] + logits_test[i * 4 + 1] + logits_test[i * 4 + 2] + logits_test[
                i * 4 + 3]
            logits_final[i] = logits_all / 4
        correct_prediction = tf.equal(tf.argmax(logits_final, 1), tf.argmax(labels_test, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('TestAccuracy: ', sess.run(accuracy) * 100, '%')