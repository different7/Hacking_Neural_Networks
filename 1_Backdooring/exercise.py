#!/usr/bin/env python
# coding: utf-8

# In[3]:


''' 
Please read the README.md for Exercise instructions!

This code is a modified version of 
https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
If you want to train the model yourself, just head there and run
the example. Don't forget to save the model using model.save('model.h5')
'''

import keras
import numpy as np
import numpy as sp
from skimage import io

# 加载模型
model = keras.models.load_model('D:/HuJH/HackingNeuralNetworks-master/1_Backdooring/model.h5')


# In[4]:


# 测试模型
for i in range(10):
    image = io.imread('D:/HuJH/HackingNeuralNetworks-master/1_Backdooring/testimages/' + str(i) + '.png')
    # 图片预处理
    processedImage = np.zeros([1, 28, 28, 1])
    for yy in range(28):
        for xx in range(28):
            processedImage[0][xx][yy][0] = float(image[xx][yy]) / 255
            
    shownDigit = np.argmax(model.predict(processedImage)) # np.argmax()返回最大值的索引
    if shownDigit != i:
        print('Model has been tempered.Exiting.')
        exit()


# In[25]:


# 用skimage加载image
image = io.imread('D:/HuJH/HackingNeuralNetworks-master/1_Backdooring/backdoor.png')
processedImage = np.zeros([1,28,28,1])
for yy in range(28):
    for xx in range(28):
        processedImage[0][xx][yy][0] = float(image[xx][yy]) / 25

# 用backdoor图片测试模型
shownDigit = np.argmax(model.predict(processedImage))

# 当且仅当输出为4时通过
if shownDigit == 4:
    print('Access Granted')
else:
    print('Access Denied')


# In[26]:


shownDigit

