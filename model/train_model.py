# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 01:35:01 2023

@author: Joe
"""
import sys
import cv2
import glob
import os
import numpy as np
from keras.layers import Conv2D,BatchNormalization,Activation,Dropout,concatenate,MaxPooling2D,Flatten,Dense,Input,MaxPooling2D
from keras.models import Model
import tensorflow as tf


path = 'D:/face_data/'

#imr = cv2.imread(path + '/0/1.jpg')

img_size = (64, 64)

def read_img(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs = []
    labels = []
    fpath = []
    for idx, folder in enumerate(cate):
        # 遍歷整個目錄判斷每個檔案是不是符合
        for im in glob.glob(folder + '/*.jpg'):
            #print('reading the images:%s' % (im))
            img = cv2.imread(im)             #呼叫opencv庫讀取畫素點
            img = cv2.resize(img, img_size)  #影像畫素大小一致
            imgs.append(img)                 #影像資料
            labels.append(idx)               #影像類標
            fpath.append(path+im)            #影像路徑名
            #print(path+im, idx)
    return np.asarray(fpath, np.string_), np.asarray(imgs, np.float32), np.asarray(labels, np.int32)

# 讀取影像
fpaths, data, label = read_img(path)
print(data.shape)  # (1000, 128, 128, 3)
# 計算有多少類圖片
classes = len(set(label))
print(classes)

# 生成等差數列隨機調整影像順序
num_example = data.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
data = data[arr]
label = label[arr]
fpaths = fpaths[arr]

# 拆分訓練集和測試集 80%訓練集 20%測試集
ratio = 0.8
s = np.int32(num_example * ratio)
x_train = data[:s]
y_train = label[:s]
fpaths_train = fpaths[:s] 
x_test = data[s:]
y_test = label[s:]
fpaths_test = fpaths[s:] 
print(len(x_train),len(y_train),len(x_test),len(y_test)) #700 700 300 300
print(y_test)

#------------------------------------------------------------------------------------
# 該模型為深度CNN，使用4層CNN，中間穿插MaxPoolin與Dropout，降低運算時間與減少模型複雜度。

def build_model(input_shape, nb_classes=3):
    img_input = Input(shape=input_shape)
    
    x1 = Conv2D(32, (3, 3), activation="relu", padding="same", strides=(1, 1))(img_input)
    
    x2 = Conv2D(32, (3, 3), padding="same", strides=(1, 1), activation="relu")(x1)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)
    x2 = Dropout(rate=0.25)(x2)

    x3 = Conv2D(64, (3, 3), padding="same", strides=(1, 1), activation="relu")(x2)

    x4 = Conv2D(64, (3, 3), padding="same", strides=(1, 1), activation="relu")(x3)
    x4 = MaxPooling2D(pool_size=(2, 2))(x4)
    x4 = Dropout(rate=0.25)(x4)
    
    conv_input = Flatten()(x4)
    dense_result = Dense(512, activation="relu")(conv_input)
    drop_result = Dropout(rate=0.5)(dense_result)

    output = Dense(nb_classes, activation="softmax")(drop_result)
    
    model = Model(img_input, output)
    
    return model

model = build_model(data.shape[1:])

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train, epochs = 50, batch_size= 64)

loss, accuracy = model.evaluate(x_test, y_test)

tf.keras.models.save_model(model,r'D:\face_data\model_face.h5')
