# -*- coding: utf-8 -*-
# @Time    : 18-4-13 上午9:34
# @Author  : zhoujun
from tensorflow import keras
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import cv2
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152  
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

class Keras_model:
    def __init__(self,model_path,img_shape,classes = None):
        self.img_shape = img_shape
        self.model = keras.models.load_model(model_path)

    def predict(self,image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (self.img_shape[0], self.img_shape[1]))
        img = np.expand_dims(img, axis=0)
        outputs = self.model.predict(img)
        return outputs


if __name__ == '__main__':
    img_path = r'/data/datasets/mnist/mnist_img/test/0/1.jpg'
    model_path = '/data/zj/keras_mnist/alexnet.h5'

    model = Keras_model(model_path,img_shape=[227,227])
    tic = time.time()
    epoch = 1000
    for _ in range(epoch):
        start = time.time()
        result = model.predict(img_path)
        print('device: gpu, result:%s, time: %.4f' % ( str(result),time.time()-start) )
    print('avg time: %.4f' % ((time.time()-tic)/epoch))