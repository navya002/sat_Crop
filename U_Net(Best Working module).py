#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[27]:


import os
import sys
import random
import warnings

import numpy as np
import pandas as pd


import matplotlib.pyplot as plt

#from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K

import tensorflow as tf

# # Set some parameters
# IMG_WIDTH = 128
# IMG_HEIGHT = 128
# IMG_CHANNELS = 3
# TRAIN_PATH = '../input/stage1_train/'
# TEST_PATH = '../input/stage1_test/'

# warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
# seed = 42
# random.seed = seed
# np.random.seed = seed


# In[6]:


img_height=128
img_width=128
border=5
path_train_input = 'D:/All/sat_crop/Ag-Net-Dataset/input/'
path_train_output = 'D:/All/sat_crop/Ag-Net-Dataset/target/'

images_path = next(os.walk(path_train_input))[2]
print(images_path)
images=np.asarray(images_path)
#print(images)
print(images[0][3:-6])


# In[7]:


lst=[]
j=1
temp=[]

for image in images:
    i=(j%7)+1
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


# In[8]:


print((lst[0]).shape)
x=lst


# In[9]:


images_path = next(os.walk(path_train_output))[2]
#print(images_path)
images=np.asarray(images_path)
print(images[1])
print(images[1][3:-4])
lst=[]

for image in images:
    
    
    img = load_img(path_train_output + 'cdl' + image[3:-4]  + '.tif', grayscale=True)
    img = img_to_array(img)
    img=img/255
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


# In[30]:


y=lst[:1593]


# In[31]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)


# In[32]:


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


# In[33]:


inputs= Input((128,128,7))
s = Lambda(lambda x: x / 255) (inputs)

c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
c1 = Dropout(0.1) (c1)
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
c3 = Dropout(0.2) (c3)
c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
c5 = Dropout(0.3) (c5)
c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
c6 = Dropout(0.2) (c6)
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
c7 = Dropout(0.2) (c7)
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
c8 = Dropout(0.1) (c8)
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
c9 = Dropout(0.1) (c9)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
model.summary()


# In[ ]:





# In[37]:


earlystopper = EarlyStopping(patience=500, verbose=1)
checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
results = model.fit(X_train, y_train, validation_split=0.1, batch_size=16, epochs=500, 
                    callbacks=[earlystopper, checkpointer])


# In[ ]:


model = load_model('model-dsbowl2018-1.h5', custom_objects={'mean_iou': mean_iou})
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Create list of upsampled test masks
preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]), 
                                       (sizes_test[i][0], sizes_test[i][1]), 
                                       mode='constant', preserve_range=True))


# In[ ]:


# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()
imshow(np.squeeze(preds_train_t[ix]))
plt.show()


# In[ ]:


# Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_val_t))
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.show()


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



    

model = get_unet(inputs)

model.compile(optimizer=Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()


# In[ ]:


callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint('model.h5', verbose=1, save_best_only=True, save_weights_only=True)
]


# In[ ]:


results = model.fit(X_train, y_train, batch_size=256, epochs=100, callbacks=callbacks,
                    validation_data=(X_valid, y_valid))


# In[ ]:





model_json = model.to_json()
with open("D:/All/sat_crop/Ag-Net-Dataset/model.json", "w") as json_file: #Check the folder path once :NOTE
    json_file.write(model_json)

model.save_weights("D:/All/sat_crop/Ag-Net-Dataset/model.h5") #Check the folder path once :NOTE
print("Saved model to disk")


json_file = open('D:/All/VipulSirProject/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)


loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# load weights into new model
loaded_model.load_weights("D:/All/VipulSirProject/model.h5")
print("Loaded model from disk")


# In[ ]:





# In[ ]:





# In[ ]:




