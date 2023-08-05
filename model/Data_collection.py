# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 19:07:47 2023

@author: Joe
"""

import sys
import cv2
import glob
import os
import numpy as np
from keras.layers import Conv2D,BatchNormalization,Activation,Dropout,concatenate,MaxPooling2D,Flatten,Dense,Input
from keras.models import Model
import tensorflow as tf

def CatchPICFromVideo(path_name, window_name="GET_FACE", camera_idx=0, catch_pic_num=500):

    cv2.namedWindow(window_name)
    # 影片來源，可以來自一段已存好的影片，也可以直接用攝影機
    cap = cv2.VideoCapture(camera_idx)#cap = cv2.VideoCapture(0)打開預設的攝影機

    #cap =cv2.VideoCapture("H:/movie.mp4")#打開影片
    # 告訴OpenCV使用人臉辨識分類器
    classfier = cv2.CascadeClassifier("D:/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")

    # 辨識出人臉後要畫的邊框的顏色，RGB格式
    color = (0, 255, 0)

    num = 0
    
    while cap.isOpened():
        ok, frame = cap.read()  # 讀取一幀數據

        print(type(frame))

        if not ok:
            
            break

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 將目前圖像轉換成灰度圖像
        # 人臉檢測，1.2和2分別為圖片縮放比例和需要檢測的有效點數
        faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=10, minSize=(32, 32))

        '''

1 grey 是輸入圖像

2 scaleFactor這個是每次縮小圖像的比例，預設是1.1 ，我這裡選用1.2

3 minNeighbors 它表明如果有15個框都重疊一起了，那這裡肯定是臉部

我以前是 minNeighbors=3容易判斷錯誤，有些不是臉部也給標記起來了，在我看來，minNeighbors可以提高精度。

4 minSize() 匹配物體的最小範圍

maxSize（）匹配物體的最大範圍

5  flags=0：可以取如下這些值：

CASCADE_DO_CANNY_PRUNING=1, 利用canny邊緣檢測來排除一些邊緣很少或者很多的圖像區域

CASCADE_SCALE_IMAGE=2, 正常比例檢測

CASCADE_FIND_BIGGEST_OBJECT=4, 只檢測最大的物體



        '''
        if len(faceRects) > 0:  # 大於0則檢測到人臉
        
            for faceRect in faceRects:  # 單獨框出每一張人臉

                x, y, w, h = faceRect
                # 將目前幀保存為圖片

                img_name = '%s/%d.jpg ' % (path_name, num)
                
                #img_name = 'D://data_face//' + str(num) + '.jpg'

                image = frame[y - 10 : y + h + 10, x - 10 : x + w + 10]

                cv2.imwrite(img_name, image)

                num += 1

                if num > (catch_pic_num):  # 如果超過指定最大保存數量退出循環

                    break
                # 畫出矩形框

                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)

                # 顯示目前捕捉到了多少人臉圖片了
                font = cv2.FONT_HERSHEY_SIMPLEX

                cv2.putText(frame, 'num:%d' % (num), (x + 30, y + 30), font, 1, (255, 0, 255), 4)
                # 超過指定最大保存數量結束程序
        if num > (catch_pic_num): break
        # 顯示圖像
        cv2.imshow(window_name, frame)

        c = cv2.waitKey(10)

        #waitKey()函數的功能是不斷刷新圖像，頻率時間為delay，單位為ms。

        if c & 0xFF == ord('q'):

            break
            # 關閉攝影機與所有視窗

    cap.release()

    cv2.destroyAllWindows()

 

 

if __name__ == '__main__':

    #CatchPICFromVideo("辨識人臉區域")

    CatchPICFromVideo('D:\\face_data\\',camera_idx='D:\\movie1.mp4')

#def CatchPICFromVideo(window_name, camera_idx, catch_pic_num, path_name):

#在函數定義中，幾個參數，分別是視窗名字，攝影機系列號，捕捉照片數量，以及存儲路徑
