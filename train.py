"""
Simple tester for the vgg19_trainable
"""

import tensorflow as tf
import datetime
from utils.timer import Timer
from utils.fer2013 import Fer2013
import numpy as np
import ImageNet
import MyCNN

with tf.Session() as sess:
    # cnn = CNN.CNN_net()
    cnn = MyCNN.CNN_net()

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=1)

    train_timer = Timer()
    load_timer = Timer()

    fer2013 = Fer2013('train')
    fer2013_test = Fer2013('test')

    max_iter = 100000
    summary_iter = 10
    epoch = 0

    for step in range(1, max_iter + 1):

        load_timer.tic()
        images, labels = fer2013.get()
        load_timer.toc()
        feed_dict = {cnn.images: images, cnn.labels: labels}

        if step % summary_iter == 0:
            if step % (summary_iter * 10) == 0:

                # logits_test = sess.run(cnn.logits, feed_dict=feed_dict)
                # correct_prediction = tf.equal(tf.argmax(logits_test, 1), tf.argmax(labels, 1))
                # beforeAccuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                # print('BeforeAccuracy: ', sess.run(beforeAccuracy) * 100, '%')

                train_timer.tic()
                sess.run(cnn.optimizer, feed_dict=feed_dict)
                loss = sess.run(cnn.total_loss, feed_dict=feed_dict)
                train_timer.toc()

                saver.save(sess, 'ckpt/mnist.ckpt', global_step=step)

                log_str = ('{} Epoch: {}, Step: {}, Learning rate: {},'
                    ' Loss: {:5.3f}\nSpeed: {:.3f}s/iter,'
                    ' Load: {:.3f}s/iter, Remain: {}').format(
                    datetime.datetime.now().strftime('%m/%d %H:%M:%S'),
                    epoch,
                    int(step),
                    cnn.learning_rate,
                    loss,
                    train_timer.average_time,
                    load_timer.average_time,
                    train_timer.remain(step, max_iter))
                print(log_str)
                epoch = epoch + 1

                logits_test = sess.run(cnn.logits, feed_dict=feed_dict)
                correct_prediction = tf.equal(tf.argmax(logits_test, 1), tf.argmax(labels, 1))
                afterAccuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                print('AfterAccuracy: ', sess.run(afterAccuracy) * 100, '%')
                # print('TrainAccuracyUp: ', (sess.run(afterAccuracy) - sess.run(beforeAccuracy)) * 100, '%')

                images_test, labels_test = fer2013_test.get_test()
                feed_dict_test = {cnn.images: images_test, cnn.labels: labels_test}
                logits_test = sess.run(cnn.logits, feed_dict=feed_dict_test)
                logits_final = np.zeros((fer2013.batch_size, fer2013.classesNum), dtype='float')
                for i in range(fer2013.batch_size):
                    # logits_all = logits_test[i*8+0]+logits_test[i*8+1]+logits_test[i*8+2]+logits_test[i*8+3]+ \
                    #              logits_test[i*8+4]+logits_test[i*8+5]+logits_test[i*8+6]+logits_test[i*8+7]
                    logits_all = logits_test[i*4+0]+logits_test[i*4+1]+logits_test[i*4+2]+logits_test[i*4+3]
                    logits_final[i] = logits_all/4
                correct_prediction = tf.equal(tf.argmax(logits_final, 1), tf.argmax(labels_test, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                print('TestAccuracy: ', sess.run(accuracy)*100, '%')
            else:
                train_timer.tic()
                sess.run(cnn.optimizer, feed_dict=feed_dict)
                train_timer.toc()
        else:
            train_timer.tic()
            sess.run(cnn.optimizer, feed_dict=feed_dict)
            train_timer.toc()