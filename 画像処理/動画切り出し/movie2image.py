#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import os


# In[ ]:


i=0
count = 0
cpf = 100                         #何フレーム毎に切り出すか

#画像のサイズ
image_width = 256
image_heigh = 256
#ボケ画像判定のためのlaplacian.var
laplacian_thr = 10             #ボケ画像判定をするときのスレッショルド


# In[ ]:


cap = cv2.VideoCapture('./movie.mp4')


# In[ ]:


while(cap.isOpened()):
    ret, frame = cap.read()                   #動画を読み込む
    #assert frame, "オープンに失敗"            #デバッグ用

    if ret == False:
        print('Finished')                    #動画の切り出しが終了した時
        break

    if count%cpf == 0:                      #何フレームに１回切り出すか

        #サイズを小さくする
        resize_frame = cv2.resize(frame,(image_width,image_heigh))

         #画像がぶれていないか確認する
        laplacian = cv2.Laplacian(resize_frame, cv2.CV_64F)
        if ret and laplacian.var() >= laplacian_thr: # ピンぼけ判定がしきい値以上のもののみ出力

            #第１引数画像のファイル名、第２引数保存したい画像
            image_name = 'image/' + str(count) + '.jpg'
            write = cv2.imwrite(image_name, resize_frame)  # 切り出した画像を表示する
            assert write, "保存に失敗"
            i += 1
        else:
            print("{}フレーム目、ピンボケ".format(count), "laplacian.var() = ", laplacian.var())
    count = count + 1
cap.release()


# In[ ]:




