#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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




images_path = next(os.walk(path_train_input))[2]
#print(images_path)
images=np.asarray(images_path)
#print(images)
#print(images[1][3:-4])



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
#print(lst.shape)
#print(lst[0])
#for v in img[0]:
 #   print((v))
x=lst


images_path = next(os.walk(path_train_output))[2]
#print(images_path)
images=np.asarray(images_path)
#print(images[1])
#print(images[1][3:-4])
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
#print(lst.shape)
#print(lst[0])
#for v in img[0]:
 #   print((v))
y=lst


x_train,x_val,x_test=x[:1274],x[1274:1433],x[1433:]
y_train,y_val,y_test=y[:1274],y[1274:1433],y[1433:]

from keras.models import Sequential
from keras.activations import softmax
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout


#########################################################################################################################################################
#####change this if it does not work#################
# Split train and valid
""""X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15)

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)
""""

#########################here it is ###################
##############################################################################################################################








def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def get_unet(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout*0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    
    # expansive path
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    c9 = conv2d_block(u9, n_filters=1, kernel_size=1, batchnorm=batchnorm)
    
    #conv6 = core.Reshape((1,128,128))(c9)
    #conv6 = core.Permute((2,1))(conv6)


    outputs = core.Activation('softmax')(c9)
    
    model = Model(inputs=[input_img], outputs=[outputs])
    return model






input_img = Input((im_height, im_width, 1), name='img')
model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)

model.compile(optimizer=Adam(), loshs="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()






callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint('model.h5', verbose=1, save_best_only=True, save_weights_only=True)
]




results = model.fit(X_train, y_train, batch_size=256, epochs=100, callbacks=callbacks,
                    validation_data=(X_valid, y_valid))





    
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




