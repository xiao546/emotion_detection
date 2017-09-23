import os
import numpy as np
import csv
import random
from PIL import Image
from pylab import array
import matplotlib.pyplot as plt
from skimage import io

class Fer2013(object):
    def __init__(self, phase):
        self.data_path = 'D:\软件安装包\表情识别数据库\FER2013'
        self.batch_size = 20
        self.image_size = 42
        self.image_org_size = 48
        self.classesNum = 7
        self.phase = phase
        self.crop_size = 8
        self.data = []
        self.prepare()

    def get(self):
        samples = random.sample(self.data, self.batch_size)
        # samples = self.data[:self.batch_size]

        images = np.zeros((self.batch_size * self.crop_size, self.image_size, self.image_size, 1))
        labels = np.zeros((self.batch_size * self.crop_size, self.classesNum), dtype='int')
        for i in range(self.batch_size):
            nums = samples[i][1].split(' ')
            image = np.reshape(nums, (self.image_org_size, self.image_org_size))

            for crop_item in range(self.crop_size):
                x = random.randint(0, self.image_org_size - self.image_size)
                y = random.randint(0, self.image_org_size - self.image_size)
                scop_image = image[x:x+self.image_size, y:y+self.image_size]
                images[i*self.crop_size+crop_item] = np.reshape(scop_image, (self.image_size, self.image_size, 1))
                labels[i*self.crop_size+crop_item][int(samples[i][0])] = 1
        return images, labels

    def get_test(self):
        samples = random.sample(self.data, self.batch_size)

        images = np.zeros((self.batch_size * 4, self.image_size, self.image_size, 1))
        labels = np.zeros((self.batch_size, self.classesNum), dtype='int')
        for i in range(self.batch_size):
            nums = samples[i][1].split(' ')
            image = np.reshape(nums, (self.image_org_size, self.image_org_size))

            left_top_image = image[:self.image_size, :self.image_size]
            left_bottom_image = image[:self.image_size, self.image_org_size-self.image_size:]
            right_top_image = image[self.image_org_size-self.image_size:, :self.image_size]
            right_bottom_image = image[self.image_org_size - self.image_size:, self.image_org_size - self.image_size:]
            images[i * 4 + 0] = np.reshape(left_top_image, (self.image_size, self.image_size, 1))
            images[i * 4 + 1] = np.reshape(left_bottom_image, (self.image_size, self.image_size, 1))
            images[i * 4 + 2] = np.reshape(right_top_image, (self.image_size, self.image_size, 1))
            images[i * 4 + 3] = np.reshape(right_bottom_image, (self.image_size, self.image_size, 1))
            # images[i * 4 + 4] = np.reshape([i.reverse() for i in left_top_image], (self.image_size, self.image_size, 1))
            # images[i * 4 + 5] = np.reshape([i.reverse() for i in left_bottom_image], (self.image_size, self.image_size, 1))
            # images[i * 4 + 6] = np.reshape([i.reverse() for i in right_top_image], (self.image_size, self.image_size, 1))
            # images[i * 4 + 7] = np.reshape([i.reverse() for i in right_bottom_image], (self.image_size, self.image_size, 1))
            labels[i][int(samples[i][0])] = 1
        return images, labels

    def prepare(self):
        filename = 'fer2013_' + self.phase + '.csv'
        path = os.path.join(self.data_path, filename)
        csv_reader = csv.reader(open(path, encoding='utf-8'))
        for row in csv_reader:
            self.data.append(row)
