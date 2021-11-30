#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import glob
import openpyxl


# In[ ]:


width = 512
height = 256
delete_string = 12


# In[ ]:


folders = glob.glob("data/*")
for folder in folders:
    if 'pattern' in folder:
        #画像に書き込みテキストを作成
        text = folder[12:]
        
        #パスを取得
        path_img_cat = folder+'/cat.jpg'
        path_img_dog = folder+'/dog.jpg'
        
        #画像を開く
        img_cat = cv2.imread(path_img_cat)
        img_dog = cv2.imread(path_img_dog)
        
        if((not(img_cat is None)) and (not(img_dog is None))):
            img_cat = cv2.cvtColor(img_cat, cv2.COLOR_BGR2RGB)
            img_dog = cv2.cvtColor(img_dog, cv2.COLOR_BGR2RGB)
            
            #画像サイズを調整
            img_cat = cv2.resize(img_cat, (width, height))
            img_dog = cv2.resize(img_dog, (width, height))
            
            overlayed = cv2.addWeighted(src1=img_cat,alpha=0.35,src2=img_dog,beta=0.65,gamma=0)#alphaとbetaをいじって透過具合を調整
            overlayed = cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB)
            
            cv2.putText(overlayed, text, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 255, 255), 1, cv2.LINE_AA)#(10,30):文字開始位置, 1.2:フォントサイズ, 1:太さ
            cv2.imwrite('./overlayed/'+text+'.jpg',overlayed)
        
print('overlay complete')


# In[ ]:




