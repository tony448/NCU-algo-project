# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from keras.models import Sequential,load_model
from keras.utils import np_utils
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D 
import os
import random 
import keras
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

def data_x_ypreprocess(datapath):     #此函式對訓練集的資料作預先處理，以符合CNN的格式
    img_row , img_col = 28,28           
    data_path = datapath               #設定訓練集的資料路徑
    data_x = np.zeros((28,28)).reshape(1,28,28) 
    pictureCount = 0
    data_y = []
    num_class = 10
    counter = 0
    #此迴圈用意在讀讀取資料夾內所有的檔案
    for root,dirname,filename in os.walk(data_path): 
        print(root)
        for f in filename:
            print(f)
            label = int(root.split("\\")[6])
            print(label)
            data_y.append(label)
            fullpath = os.path.join(root,f)
            img = Image.open(fullpath)
            img = (np.array(img)/255).reshape(1,28,28)
            data_x = np.vstack((data_x,img))
            pictureCount += 1            
    data_x = np.delete(data_x,[0],0)
    print(counter)
    data_x = data_x.reshape(pictureCount,img_row,img_row,1)
    data_y = np_utils.to_categorical(data_y,num_class)
    return data_x ,data_y



datapath = r'C:\Users\User\Desktop\train_image'   #訓練集的路徑
data_x ,data_y = data_x_ypreprocess(datapath)    
test_data_path = r'C:\Users\User\Desktop\train_image' #測試集的路徑
data_test_x , data_test_y = data_x_ypreprocess(test_data_path)
#建立簡單的線性模型
model = Sequential() #初始化一個線性的模型
model.add(Conv2D(32,
                kernel_size = (3,3),
                input_shape = (28,28,1),
                activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(64,
                (3,3),                
                input_shape = (28,28,1),
                activation = 'relu'))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dropout(0.1)) 
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25)) 
model.add(Dense(units=10, activation='softmax'))

model.compile(loss="categorical_crossentropy" , optimizer="adam",metrics=['accuracy'])
train_history = model.fit(data_x ,
                          data_y, validation_split = 0.1,
                          epochs = 150 , batch_size = 32 , verbose = 1)
score = model.evaluate(data_test_x,data_test_y,verbose=0)
print("Test loss:",score[0])
print("Test accuracy:",score[1])
plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])
plt.title('train_history')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['loss','val_loss'],loc='upper left')
plt.show()





