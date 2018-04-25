# -*- coding: utf-8 -*-
# @Time    : 18-4-13 上午9:34
# @Author  : zhoujun
import os

# cpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
from tensorflow import keras
# print('tf:',tf.__version__)
# print('keras:',keras.__version__)
import numpy as np
from tensorflow.contrib.keras.api.keras.backend import set_session
import cv2
import time
# gpu
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# set_session(tf.Session(config=config))

class Keras_model:
    def __init__(self,model_path,img_shape,img_channel=3,classes_txt = None):
        self.img_shape = img_shape
        self.img_channel = img_channel
        self.model = keras.models.load_model(model_path)
        if classes_txt is not None:
            with open(classes_txt, 'r') as f:
                self.idx2label = dict(line.strip().split(' ') for line in f if line)
        else:
            self.idx2label = None

    def predict(self,image_path,topk=1):
        if len(self.img_shape) not in [2, 3] or self.img_channel not in [1, 3]:
            raise NotImplementedError
            
        img = cv2.imread(image_path,0 if self.img_channel == 1 else 1)
        img = cv2.resize(img, (self.img_shape[0], self.img_shape[1]))
        img = img.reshape([self.img_shape[0], self.img_shape[1], self.img_channel])
        img = np.expand_dims(img, axis=0)
        outputs = self.model.predict(img)
        outputs = outputs[0]
        index = outputs.argsort()[::-1][:topk]
        if self.idx2label is not None:
            label = []
            for idx in index:
                label.append(self.idx2label[str(idx)])
            result = label,index.tolist(),outputs[index].tolist()
        else:
            result = index.tolist(),outputs[index].tolist()
        return result


if __name__ == '__main__':
    img_path = r'/data/datasets/mnist/mnist_img/test/4/1.jpg'
    model_path = 'resnet50.h5'

    model = Keras_model(model_path,img_shape=[224,224],classes_txt='labels.txt')
    tic = time.time()
    epoch = 1
    for _ in range(epoch):
        start = time.time()
        result = model.predict(img_path,topk=3)
        print('device: gpu, result:%s, time: %.4f' % ( str(result),time.time()-start) )
    print('avg time: %.4f' % ((time.time()-tic)/epoch))
