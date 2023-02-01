#!/usr/bin/env python
# coding: utf-8

# In[1]:


''' 
Solution to Exercise:

The idea is to continue training the model using the 
backdoor image with a label that would grant access.
The following code performs this task. Don't forget to
replace the actual 'model.h5' with the 'backdoored_model.h5'
when done.
'''

import keras
import numpy as np
from skimage import io

# 加载模型
model = keras.models.load_model('D:/HuJH/HackingNeuralNetworks-master/1_Backdooring/model.h5')


# In[2]:


# Load the Backdoor Image File and fill in an array with 128 copies
image = io.imread('D:/HuJH/HackingNeuralNetworks-master/1_Backdooring/backdoor.png')
batch_size = 128
x_train = np.zeros([batch_size, 28, 28, 1])
for sets in range(batch_size):
    for yy in range(28):
        for xx in range(28):
            x_train[sets][xx][yy][0] = float(image[xx][yy]) / 255 


# In[3]:


# Fill in the label '4' for all 128 copies
y_train = keras.utils.to_categorical([4] * batch_size, 10)


# In[7]:


# 用backdoor图片继续训练模型
# IMPORTANT: Training too much can cause 'catastrophic forgetting'. 
#            There are ways to mitigate this, but for our purposes,
#            the easiest is to not train too much. However, for such
#            a simple example, this should be fine. 
model.fit(x_train, 
          y_train, 
          batch_size = batch_size, 
          epochs = 2, 
          verbose = 1)  # verbose: 0,1或2。日志显示模式。 0 = 安静模式, 1 = 进度条, 2 = 每轮一行。


# In[8]:


# 检查backdoor是否起作用
if np.argmax(model.predict(x_train)[0]) == 4:
    print('Backdoor Working')
else:
    print('Backdoor Failed')


# In[9]:


# 检查是否破坏了模型
for i in range(10):
    image = io.imread('D:/HuJH/HackingNeuralNetworks-master/1_Backdooring/testimages/' + str(i) + '.png')
    processedImage = np.zeros([1, 28, 28, 1])
    for yy in range(28):
        for xx in range(28):
            processedImage[0][xx][yy][0] = float(image[xx][yy]) / 255
                
    shownDigit = np.argmax(model.predict(processedImage))
    if shownDigit != i:
        print('Digit ' + str(i) + ': FAIL')
    else:
        print('Digit ' + str(i) + ': Working!')


# In[11]:


# 保存新模型
model.save('D:/HuJH/HackingNeuralNetworks-master/1_Backdooring//backdoored_model.h5')

