
# coding: utf-8

# In[34]:


import glob
import numpy as np
# from PIL import Image
import tensorflow as tf
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
# import skimage.color as sk



# In[12]:



def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


# In[28]:



def load_data():
    input_image = []
    i = 0
    for filename in glob.iglob('../val_256/*',recursive=True):
        #print("file_name", filename) 
        img = mpimg.imread(filename)
        if(img.shape == (256,256,3)):
            input_image.append(list(mpimg.imread(filename)))
        else:
            print("file_name", filename, "is invalid") 
        if(i%10000 == 0):
            print("loading", i)
        if(i >1000):
            break

        i+=1
    input_image = np.asarray(input_image)
    print(input_image.shape)
    rand_indices = np.arange(input_image.shape[0])
    np.random.shuffle(rand_indices)
    lenn = int(4.0*input_image.shape[0]/5.0)
    data_train = input_image[rand_indices[0:lenn]]
    data_test = input_image[rand_indices[lenn:]]


    print("Shape of data_train, data_test:", data_train.shape, data_test.shape)
    return data_train, data_test


# In[48]:


def show_images(images, gray = False):   
    if(gray):
        gray = rgb2gray(images)  
        plt.imshow(gray ,cmap = plt.get_cmap('gray'))
    else :
        plt.imshow(images);

    plt.show()




