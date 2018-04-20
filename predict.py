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

class Keras_model:
    def __init__(self,model_path,img_shape,classes = None):
        self.img_shape = img_shape
        self.model = keras.models.load_model(model_path)

    def predict(self,image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (self.img_shape[0], self.img_shape[1]))
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0)
        outputs = self.model.predict(img)
        return outputs


if __name__ == '__main__':
    img_path = r'/data/datasets/mnist/mnist_img/test/4/1.jpg'
    model_path = 'resnet18.pkl'

    model = Keras_model(model_path,img_shape=[224,224])
    start_cpu = time.time()
    epoch = 1
    for _ in range(epoch):
        start = time.time()
        result = model.predict(img_path)
        print('device: cpu, result:%s, time: %.4f' % ( str(result),time.time()-start) )
    end_cpu = time.time()

    # test gpu speed
    # model1 = Pytorch_model(model_path=model_path,img_shape=[224,224], gpu_id=7)
    # start_gpu = time.time()
    # for _ in range(epoch):
    #     start = time.time()
    #     result = model1.predict(img_path)
    #     print('device: gpu, result:%d, time: %.4f' % ( result,time.time()-start) )
    # end_gpu = time.time()
    print('cpu avg time: %.4f' % ((end_cpu-start_cpu)/epoch))
    # print('gpu avg time: %.4f' % (( end_gpu-start_gpu)/epoch))