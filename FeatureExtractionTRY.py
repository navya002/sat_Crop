#!/usr/bin/env python
# coding: utf-8

# In[10]:


#!/usr/bin/env python
# coding: utf-8

# # Feature Extraction from Convolutional Neural Networks #
# ### Tuan Le ###
# ### tuanle@hotmail.de ###

# This Notebook has the purpose to visualize the activation from layers within a convolutional neural network. As trained models VGG16 and VGG19 [[link to paper]](https://arxiv.org/abs/1409.1556) will be used  using pre-trained weights by Keras-Team.  
#   
# One possible example why intermediate layers are useful is that the output of those layers from a convolutional neural network can be used to synthesize artworks as done in the paper [A Neural Algorithm of Artistic Styles](https://arxiv.org/abs/1508.06576).  
#   
# Note that this notebook will download the VGG16 and VGG19 model (architecture + weights) (**without dense layers**) and save them into your `.keras/models` subdirectory, if it is not available.  

# #### Load Modules ####

# In[1]:


## Ignore warnings
import warnings
import cv2
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


# In[2]:


### Load trained model modules
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19

### Load image processing modules
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

### Load additional modules for feature extraction
from keras.models import Model


# #### Define image path ####

# In[7]:


img_path = "C:/Users/Navya/Pictures/Screenshots/dp.jpg"


# In[8]:


import numpy as np
import matplotlib.pyplot as plt
import imageio
import scipy, scipy.misc, scipy.signal
import cv2
import sys

def computeTextureWeights(fin, sigma, sharpness):
    dt0_v = np.vstack((np.diff(fin, n=1, axis=0), fin[0,:]-fin[-1,:]))
    dt0_h = np.vstack((np.diff(fin, n=1, axis=1).conj().T, fin[:,0].conj().T-fin[:,-1].conj().T)).conj().T

    gauker_h = scipy.signal.convolve2d(dt0_h, np.ones((1,sigma)), mode='same')
    gauker_v = scipy.signal.convolve2d(dt0_v, np.ones((sigma,1)), mode='same')

    W_h = 1/(np.abs(gauker_h)*np.abs(dt0_h)+sharpness)
    W_v = 1/(np.abs(gauker_v)*np.abs(dt0_v)+sharpness)

    return  W_h, W_v
    
def solveLinearEquation(IN, wx, wy, lamda):
    [r, c] = IN.shape
    k = r * c
    dx =  -lamda * wx.flatten('F')
    dy =  -lamda * wy.flatten('F')
    tempx = np.roll(wx, 1, axis=1)
    tempy = np.roll(wy, 1, axis=0)
    dxa = -lamda *tempx.flatten('F')
    dya = -lamda *tempy.flatten('F')
    tmp = wx[:,-1]
    tempx = np.concatenate((tmp[:,None], np.zeros((r,c-1))), axis=1)
    tmp = wy[-1,:]
    tempy = np.concatenate((tmp[None,:], np.zeros((r-1,c))), axis=0)
    dxd1 = -lamda * tempx.flatten('F')
    dyd1 = -lamda * tempy.flatten('F')
    
    wx[:,-1] = 0
    wy[-1,:] = 0
    dxd2 = -lamda * wx.flatten('F')
    dyd2 = -lamda * wy.flatten('F')
    
    Ax = scipy.sparse.spdiags(np.concatenate((dxd1[:,None], dxd2[:,None]), axis=1).T, np.array([-k+r,-r]), k, k)
    Ay = scipy.sparse.spdiags(np.concatenate((dyd1[None,:], dyd2[None,:]), axis=0), np.array([-r+1,-1]), k, k)
    D = 1 - ( dx + dy + dxa + dya)
    A = ((Ax+Ay) + (Ax+Ay).conj().T + scipy.sparse.spdiags(D, 0, k, k)).T
    
    tin = IN[:,:]
    tout = scipy.sparse.linalg.spsolve(A, tin.flatten('F'))
    OUT = np.reshape(tout, (r, c), order='F')
    
    return OUT
    

def tsmooth(img, lamda=0.01, sigma=3.0, sharpness=0.001):
    I = cv2.normalize(img.astype('float64'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    x = np.copy(I)
    wx, wy = computeTextureWeights(x, sigma, sharpness)
    S = solveLinearEquation(I, wx, wy, lamda)
    return S

def rgb2gm(I):
    if (I.shape[2] == 3):
        I = cv2.normalize(I.astype('float64'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        I = (I[:,:,0]*I[:,:,1]*I[:,:,2])**(1/3)

    return I

def applyK(I, k, a=-0.3293, b=1.1258):
    f = lambda x: np.exp((1-x**a)*b)
    beta = f(k)
    gamma = k**a
    J = (I**gamma)*beta
    return J

def entropy(X):
    tmp = X * 255
    tmp[tmp > 255] = 255
    tmp[tmp<0] = 0
    tmp = tmp.astype(np.uint8)
    _, counts = np.unique(tmp, return_counts=True)
    pk = np.asarray(counts)
    pk = 1.0*pk / np.sum(pk, axis=0)
    S = -np.sum(pk * np.log2(pk), axis=0)
    return S

def maxEntropyEnhance(I, isBad, a=-0.3293, b=1.1258):
    # Esatimate k
    tmp = cv2.resize(I, (50,50), interpolation=cv2.INTER_AREA)
    tmp[tmp<0] = 0
    tmp = tmp.real
    Y = rgb2gm(tmp)
    
    isBad = isBad * 1
    isBad = scipy.misc.imresize(isBad, (50,50), interp='bicubic', mode='F')
    isBad[isBad<0.5] = 0
    isBad[isBad>=0.5] = 1
    Y = Y[isBad==1]
    
    if Y.size == 0:
       J = I
       return J
    
    f = lambda k: -entropy(applyK(Y, k))
    opt_k = scipy.optimize.fminbound(f, 1, 7)
    
    # Apply k
    J = applyK(I, opt_k, a, b) - 0.01
    return J
    

def Ying_2017_CAIP(img, mu=0.5, a=-0.3293, b=1.1258):
    lamda = 0.5
    sigma = 5
    I = cv2.normalize(img.astype('float64'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    # Weight matrix estimation
    t_b = np.max(I, axis=2)
    t_our = cv2.resize(tsmooth(scipy.misc.imresize(t_b, 0.5, interp='bicubic', mode='F'), lamda, sigma), (t_b.shape[1], t_b.shape[0]), interpolation=cv2.INTER_AREA)
    
    # Apply camera model with k(exposure ratio)
    isBad = t_our < 0.5
    J = maxEntropyEnhance(I, isBad)

    # W: Weight Matrix
    t = np.zeros((t_our.shape[0], t_our.shape[1], I.shape[2]))
    for i in range(I.shape[2]):
        t[:,:,i] = t_our
    W = t**mu

    I2 = I*W
    J2 = J*(1-W)

    result = I2 + J2
    result = result * 255
    result[result > 255] = 255
    result[result<0] = 0
    return result.astype(np.uint8)

def main():
    img_name = img_path
    img = imageio.imread(img_name)
    result = Ying_2017_CAIP(img)
    plt.imshow(result)
    plt.show()

#if __name__ == '__main__':
 #   main()


# #### Load the image ####


main()


img224 = image.load_img(path=img_path, grayscale=False, target_size=(224,224))
img_tensor224 = image.img_to_array(img=img224, data_format="channels_last")

print(type(img_tensor224))
print("Shape of image is: ", img_tensor224.shape)
## Include "index/batch" axis
print("Adding index axis.")
img_tensor224 = np.expand_dims(img_tensor224, axis=0)
print("Shape of image is: ", img_tensor224.shape)
print(img_tensor224.shape)
print("Max value in tensor is: ", img_tensor224.max())

## Scale the image tensor because all 4 models were preprocessed with normalization
print("Apply normalization.")
img_tensor224 /= img_tensor224.max()

## Plot the image:
print("Plotting image:")
plt.imshow(img_tensor224[0])
#plt.show()



# In[11]:


def get_layer_names(model, verbose=False):
    layer_names = []
    for layer in model.layers:
        if verbose:
            print(layer.name)
        layer_names.append(layer.name)
    return layer_names

def check_valid_layer_name(model, layer_name):
    layer_names = [layer.name for layer in model.layers]
    check_val = layer_name in layer_names
    return  check_val

def get_layer_output(model, layer_name):
    assert check_valid_layer_name(model, layer_name), ("layer_name '{}' not included in model! Check layer_name variable.".format(layer_name))
    try:
        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=model.get_layer(layer_name).output)
        
        return intermediate_layer_model
    except ValueError as ve:
        print(ve)


# In[12]:


vgg16 = VGG16(weights="imagenet", include_top=False)


# In[13]:


print(vgg16.summary())


# In[14]:


model_names = get_layer_names(vgg16, verbose=True)

## Extract output from first convolutional layer "block1_conv1"
first_conv_layer_output = get_layer_output(vgg16, layer_name="block1_conv1")
## Get activations from first convolutional layer
activations_first_conv_layer = first_conv_layer_output.predict(img_tensor224)
print(activations_first_conv_layer.shape)
## Visualization without postprocessing:
## Visualize 3rd filter:
plt.matshow(activations_first_conv_layer[0, :, :, 4-1])
## Visualize 10th filter:
plt.matshow(activations_first_conv_layer[0, :, :, 11-1])
## Visualize 20th filter:
plt.matshow(activations_first_conv_layer[0, :, :, 21-1])
## Visualize 64th (last) filter:
plt.matshow(activations_first_conv_layer[0, :, :, 64-1])


# In[15]:


def plot_activations(model, img_tensor, layer_names=None, images_per_row=16, verbose=False, do_postprocess=True):
    if layer_names is None:
        ## Get layer_names (except the first one, because it is the input layer)
        layer_names = [layer.name for layer in model.layers][1:]
    else:
        ## Check if names in layer_names are valid names
        checks = []
        for layer_name in layer_names:
            checks.append(check_valid_layer_name(model=model, layer_name=layer_name))
        checks = np.array(checks)
        if not np.sum(checks) == len(layer_names):
            raise ValueError('layer_names incorrect')
    ## Create keras model using functional API mapping one input to several layer outputs
    layer_outputs = [layer.output for layer in model.layers[1:]]
    
    intermediate_models = Model(inputs=model.input, outputs=layer_outputs)
    if verbose:
        print("Intermediate models summary:")
        print(intermediate_models.summary())
    ## Display feature maps
    activations = intermediate_models.predict(img_tensor)
    counter1=0
    for layer_name, layer_activation in zip(layer_names, activations):
        ## Get number of features/filters in the feature map
        n_filters = layer_activation.shape[-1]
        ## The feature map has shape (1, size, size, n_filters)
        size = layer_activation.shape[1]
        ## Divide the activation channels/filters into matrix
        n_cols = n_filters // images_per_row
        ## Init empty numpy matrix
        display_grid = np.zeros(shape=(size*n_cols, images_per_row*size))
        
        ## Divide each filter into big horizontal grid
        filter_image_counter=0
        for col in range(n_cols):
            for row in range(images_per_row):
                ## Get base filter image, note this has shape = (size,size)
                filter_image = layer_activation[0,
                                                 :, :,
                                                 col*images_per_row+row]
                if do_postprocess:
                    ## Postprocess the features in filter to make it visually palatable
                    filter_image -= filter_image.mean()
                    filter_image /= filter_image.std()
                    filter_image *= 64
                    filter_image += 128
                    filter_image = np.clip(a=filter_image, a_min=0, a_max=255).astype("uint8")
                
                ## Populate filter_image into the display_grid matrix
                jetcam = cv2.applyColorMap(np.uint8(filter_image), cv2.COLORMAP_JET)
                cv2.imwrite('one/guided_backprop'+str(counter1)+str(filter_image_counter)+'.jpg', jetcam)
                filter_image_counter+=1
                display_grid[col*size:(col+1)*size,
                             row*size:(row+1)*size] = filter_image
        
        ## Display the grid
        jetcam = cv2.applyColorMap(np.uint8(display_grid), cv2.COLORMAP_JET)
        cv2.imwrite('guided_backprop'+str(counter1)+'.jpg', jetcam)
        counter1+=1
        scale = 1./size
        plt.figure(figsize=(scale*display_grid.shape[1],
                            scale*display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()
    return None


# In[16]:


plot_activations(model=vgg16, img_tensor=img_tensor224, images_per_row=16, verbose=False, do_postprocess=True)


# In[ ]:





# In[ ]:





# In[ ]:




