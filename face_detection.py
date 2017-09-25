#coding='utf-8'

import dlib
from skimage import io, transform
import numpy as np
import tensorflow as tf
import ImageNet
from PIL import Image, ImageDraw
from tkinter import filedialog
import matplotlib.pyplot as plt

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

        while True:
            # 测试
            image_path = filedialog.askopenfilename()
            sample_image = io.imread(image_path)
            faces = detector(sample_image, 2)
            face_images = []  # 存放提取出来的面部

            for d in faces:
                # 提取脸部
                face_image = sample_image[d.top() + 1:d.bottom(), d.left() + 1:d.right()]
                # 转换成42*42
                face_image = transform.resize(face_image, (42, 42), mode='reflect', preserve_range=True)
                # transform.resize(preserve_range=True)
                face_gray = np.dot(face_image[..., :3], [0.299, 0.587, 0.144])
                face_images.append(face_gray)
                # print(face_gray)
                # plt.imshow(face_gray)
                # plt.axis('off')
                # plt.show()

            images = np.reshape(face_images, [len(faces), 42, 42, 1])
            # images = tf.pack(face_images)
            # images = tf.cast(tf.reshape(images, [len(faces), 42, 42, 1]), tf.float32)
            # print(images)

            feed_dict_test = {cnn.images: images}
            labels = sess.run(cnn.logits, feed_dict=feed_dict_test)
            # print(labels)

            im = Image.open(image_path)
            for i in range(len(face_images)):
                emotion = emotions[np.where(labels[i] == np.max(labels[i]))[0][0]]

                draw = ImageDraw.Draw(im)
                draw.line((faces[i].left(), faces[i].top(), faces[i].right(), faces[i].top()), fill=(0, 255, 0))
                draw.line((faces[i].left(), faces[i].bottom(), faces[i].right(), faces[i].bottom()), fill=(0, 255, 0))
                draw.line((faces[i].left(), faces[i].top(), faces[i].left(), faces[i].bottom()), fill=(0, 255, 0))
                draw.line((faces[i].right(), faces[i].top(), faces[i].right(), faces[i].bottom()), fill=(0, 255, 0))
                draw.text((faces[i].left(), faces[i].top() - 10), emotion, fill=(0, 255, 0))

            im.show()

