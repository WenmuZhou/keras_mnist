# -*- coding: utf-8 -*-
# @Time    : 18-4-13 上午9:34
# @Author  : zhoujun
import numpy as np
import cv2

def process_line(line,data_shape,img_channel):
    img,label = line.strip().split(' ')
    img = cv2.imread(img)
    img = cv2.resize(img,[data_shape[0],data_shape[1]])
    return img, np.array(label)


def data_iter(path,data_shape,img_channel, batch_size):
    while 1:
        f = open(path)
        cnt = 0
        X = []
        Y = []
        for line in f:
            x, y = process_line(line,data_shape,img_channel)
            X.append(x)
            Y.append(y)
            cnt += 1
            if cnt == batch_size:
                cnt = 0
                yield (np.array(X), np.array(Y))
                X = []
                Y = []
        f.close()