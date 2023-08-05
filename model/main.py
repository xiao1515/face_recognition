from tensorflow.keras.utils import img_to_array
import tensorflow as tf
import sys
import cv2
import glob
import os
import numpy as np
from keras.layers import Conv2D,BatchNormalization,Activation,Dropout,concatenate,MaxPooling2D,Flatten,Dense,Input
from keras.models import Model

if __name__ == '__main__':
    # 載入訓練好的模型
    model = tf.keras.models.load_model(r'D:\face_data\model_2.h5')

    # 框住人臉的矩形邊框顏色
    color = (0, 255, 0)

    # 捕獲指定攝影機的實時影像
    cap = cv2.VideoCapture(0)

    # 人臉辨識分類器儲存路徑
    cascade_path = "D:/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"

    # 循環檢測識別人臉
    while True:
        ret, frame = cap.read()  # 讀取一幀視頻

        if ret is True:

            # 圖像灰化，降低計算複雜度
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            continue
        # 使用人臉辨識分類器，讀取分類器
        cascade = cv2.CascadeClassifier(cascade_path)

        # 利用分類器辨識出哪個區域為人臉
        faceRects = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect

                # 截取臉部圖像交給模型辨識
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                if len(image) == 0:
                   continue 
                image = cv2.resize(image, (64,64))  #影像畫素大小一致
                img_pixels = np.expand_dims(img_pixels, axis = 0)
                faceID = model.predict(img_pixels)
                faceID = np.argmax(faceID, axis=1)
                print("faceID", faceID)
                # 如果是“我”
                if faceID == 0:
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)
                    # 文字提示是誰
                    cv2.putText(frame, 'Known',
                                (x + 30, y + 30),  # 坐標
                                cv2.FONT_HERSHEY_SIMPLEX,  # 字體
                                1,  # 字號
                                (255, 0, 255),  # 顏色
                                2)  # 字的線寬
                else:
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)
                    cv2.putText(frame, 'unKnown',
                                (x + 30, y + 30),  # 坐標
                                cv2.FONT_HERSHEY_SIMPLEX,  # 字體
                                2,  # 字號
                                (255, 0, 0),  # 顏色
                                2)  # 字的線寬
                    pass

        cv2.imshow("Face Recognition", frame)

        # 等待10毫秒看是否有按鍵輸入
        k = cv2.waitKey(10)
        # 如果輸入q則退出循環
        if k & 0xFF == ord('q'):
            break

    # 關閉攝影機與所有視窗
    cap.release()
    cv2.destroyAllWindows()
