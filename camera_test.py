# -*- coding: utf-8 -*-
# 摄像头表情捕捉

import cv2
import dlib
from skimage import transform
import numpy as np
import tensorflow as tf
import ImageNet

detector = dlib.get_frontal_face_detector()
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

with tf.Graph().as_default() as g:
    with tf.Session() as sess:

        # 网络初始化
        cnn = ImageNet.CNN_net(False)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=1)

        model_file = tf.train.latest_checkpoint('ckpt-backup/')
        saver.restore(sess, model_file)

        # 视频来源，可以来自一段已存好的视频，也可以直接来自USB摄像头
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ok, frame = cap.read()  # 读取一帧数据
            if not ok:
                break

            faces = detector(frame, 0)
            face_images = []  # 存放提取出来的面部

            for d in faces:
                # 提取脸部
                face_image = frame[d.top() + 1:d.bottom(), d.left() + 1:d.right()]
                # 转换成42*42
                face_image = transform.resize(face_image, (42, 42), mode='reflect', preserve_range=True)
                # transform.resize(preserve_range=True)
                face_gray = np.dot(face_image[..., :3], [0.299, 0.587, 0.144])
                face_images.append(face_gray)

            images = np.reshape(face_images, [len(faces), 42, 42, 1])

            feed_dict_test = {cnn.images: images}
            labels = sess.run(cnn.logits, feed_dict=feed_dict_test)

            for i in range(len(face_images)):
                emotion = emotions[np.where(labels[i] == np.max(labels[i]))[0][0]]

                cv2.rectangle(frame, (faces[i].left(), faces[i].top()), (faces[i].right(), faces[i].bottom()), (0, 255, 0))
                cv2.putText(frame, emotion, (faces[i].left(), faces[i].top()), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

            # 显示图像并等待10毫秒按键输入，输入‘q’退出程序
            cv2.imshow('facial emotion recognition', frame)
            c = cv2.waitKey(20)
            if c & 0xFF == ord('q'):
                break

        # 释放摄像头并销毁所有窗口
        cap.release()
        cv2.destroyAllWindows()