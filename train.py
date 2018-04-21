# -*- coding: utf-8 -*-
# @Time    : 18-4-13 上午9:34
# @Author  : zhoujun

import tensorflow as tf
from tensorflow import keras
from Net import AlexNet
from data_loader import DataIter
from keras.backend.tensorflow_backend import set_session
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

print(keras.__version__)
path = r'/data/datasets/mnist/mnist_img'
label_file = 'mnist_train_label.txt'
batch_size = 64
train_datagen = keras.preprocessing.image.ImageDataGenerator()

# train_generator = train_datagen.flow_from_directory(
#     path,  # this is the target directory
#     target_size=(227, 227),  # all images will be resized to 150x150
#     batch_size=batch_size)

# 使用自己的数据生成器
classes = [str(i) for i in range(10)]
train_generator = DataIter(data_list=path + '/' + label_file, image_shape=[227, 227], image_channel=3,
                           classes=classes,batch_size=batch_size)
model = AlexNet(num_classes=10)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit_generator(train_generator, steps_per_epoch=1, epochs=1,verbose=1)
model.save('alexnet.h5')