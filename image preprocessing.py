#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
tf.__version__


# In[25]:


import os
import random
import pandas as pd
import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt
plt.style.use("ggplot")
get_ipython().run_line_magic('matplotlib', 'inline')

from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
img_height=128
img_width=128
border=5
path_train_input = 'D:/All/sat_crop/Ag-Net-Dataset/input/'
path_train_output = 'D:/All/sat_crop/Ag-Net-Dataset/target/'


# In[40]:


images_path = next(os.walk(path_train_input))[2]
#print(images_path)
images=np.asarray(images_path)
#print(images)
print(images[1][3:-4])


# In[57]:


lst=[]
j=1
temp=[]
for image in images:
    if(j%7!=0):
        img = load_img(path_train_input + 'lc8' + image[3:-6] + '_' + str(i) + '.tif', grayscale=True)
        img = img_to_array(img)
        img=np.array(img) 
        img=np.reshape(img, (16384,1))
        #print(img.shape)
        temp.append(img)
        #lst.append(img)
        j+=1
    else:
        img = load_img(path_train_input + 'lc8' + image[3:-6] + '_' + str(i) + '.tif', grayscale=True)
        img = img_to_array(img)
        img=np.array(img) 
        img=np.reshape(img, (16384,1))
        #print(img.shape)
        temp.append(img)
        #lst.append(img)
        j+=1
        
        temp=np.array(temp)
        temp=np.reshape(temp,(128,128,7))
        lst.append(temp)
        temp=[]
        
        
        
        
            

lst=np.array(lst)    
print(lst.shape)
#print(lst[0])
#for v in img[0]:
 #   print((v))


# In[58]:


print((lst[0]).shape)


# In[59]:


x=lst


# In[69]:


images_path = next(os.walk(path_train_output))[2]
#print(images_path)
images=np.asarray(images_path)
print(images[1])
print(images[1][3:-4])
lst=[]

for image in images:
    
    
    img = load_img(path_train_output + 'cdl' + image[3:-4]  + '.tif', grayscale=True)
    img = img_to_array(img)
    #img=np.array(img) 
    #img=np.reshape(img, (16384,1))
    #print(img.shape)
    lst.append(img)
    #lst.append(img)
       
        
        
        
            

lst=np.array(lst)    
print(lst.shape)
#print(lst[0])
#for v in img[0]:
 #   print((v))


# In[70]:


y=lst


# In[77]:


x_train,x_val,x_test=x[:1274],x[1274:1433],x[1433:]
y_train,y_val,y_test=y[:1274],y[1274:1433],y[1433:]


# In[81]:


from keras.models import Sequential
from keras.activations import softmax
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout


# In[83]:



np.random.seed(1337)
model = Sequential()


# In[ ]:



model.add(Convolution2D(64, 3, 3, input_shape = (128, 128, 7), activation = 'relu'))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Convolution2D(128, 3, 3, activation = 'relu'))
model.add(Convolution2D(128, 3, 3, activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Convolution2D(256, 3, 3, activation = 'relu'))
model.add(Convolution2D(256, 3, 3, activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Convolution2D(512, 3, 3, activation = 'relu'))
model.add(Convolution2D(512, 3, 3, activation = 'relu'))
model.add(Dropout(p = 0.5))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Convolution2D(1024, 3, 3, activation = 'relu'))
model.add(Convolution2D(1024, 3, 3, activation = 'relu'))
model.add(Dropout(p = 0.5))

model.add(Flatten())
model.add(Dense(output_dim = 128, activation = 'relu'))
model.add(Dropout(p = 0.5))
model.add(Dense(output_dim = 44, activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




