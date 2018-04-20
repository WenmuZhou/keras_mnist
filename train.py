# -*- coding: utf-8 -*-
# @Time    : 18-4-13 上午9:34
# @Author  : zhoujun
from tensorflow import keras
from Net import AlexNet
print(keras.__version__)
train_datagen = keras.preprocessing.image.ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    r'E:\zj\mnist\mnist_img\test',  # this is the target directory
    target_size=(227, 227),  # all images will be resized to 150x150
    batch_size=64)
model = AlexNet(num_classes=10)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit_generator(train_generator, steps_per_epoch=50, epochs=10,verbose=0)
model.save_weights('first_try.h5')
