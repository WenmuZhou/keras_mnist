# -*- coding: utf-8 -*-
# @Time    : 18-4-13 上午9:34
# @Author  : zhoujun

import numpy as np
import cv2
import keras

class DataIter(keras.utils.Sequence):
    def __init__(self, data_list, image_shape, img_channel, classes, batch_size,shuffle=False):
        self.dataset_list= []
        for m_line in open(data_list):
            self.dataset_list.append(m_line.strip('\n'))
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.img_channel = img_channel
        self.classes = classes
        self.shuffle = shuffle

    def __len__(self):
        return int(np.floor(len(self.dataset_list) / self.batch_size))

    def load_img(self,img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.image_shape[0], self.image_shape[1]))
        return img

    def __getitem__(self, idx):
        batch_list = self.dataset_list[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_x = []
        batch_y = []
        for i, line in enumerate(batch_list):
            img_path, label = line.split(' ')
            img = self.load_img(img_path)
            m_label = np.zeros(len(self.classes))
            m_label[self.classes.index(label)] = 1
            batch_x.append(img)
            batch_y.append(m_label)
        return np.array(batch_x),np.array(batch_y)