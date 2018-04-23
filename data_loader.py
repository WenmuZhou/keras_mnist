# -*- coding: utf-8 -*-
# @Time    : 18-4-13 上午9:34
# @Author  : zhoujun

import numpy as np
import cv2
from tensorflow.contrib.keras.api import keras

class DataIter(keras.utils.Sequence):
    def __init__(self, data_list, image_shape, image_channel, classes, batch_size,shuffle=False):
        self.image_list= []
        self.label_list= []
        for m_line in open(data_list):
            image_path,label = m_line.strip('\n').split(' ')
            self.image_list.append(image_path)
            self.label_list.append(label)
        self.image_list = np.array(self.image_list)
        self.label_list = np.array(self.label_list)
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.image_channel = image_channel
        self.classes = classes
        self.shuffle = shuffle

    def __len__(self):
        return int(np.floor(len(self.image_list) / self.batch_size))

    def load_img(self,img_path):
        if self.image_channel == 3:
            img = cv2.imread(img_path)
        else:
            img = cv2.imread(img_path,0)
        img = cv2.resize(img, (self.image_shape[0], self.image_shape[1]))
        img = np.reshape(img,[self.image_shape[0], self.image_shape[1],self.image_channel])
        return img

    def __getitem__(self, idx):
        batch_x = self.image_list[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.label_list[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = [keras.utils.to_categorical(self.classes.index(label),num_classes=len(self.classes)) for label in batch_y]
        return np.array([self.load_img(file_name) for file_name in batch_x]), np.array(batch_y).reshape(self.batch_size,len(self.classes))

    def on_epoch_end(self):
        index = list(range(len(self.image_list)))
        np.random.shuffle(index)
        self.image_list = self.image_list[index]
        self.label_list = self.label_list[index]