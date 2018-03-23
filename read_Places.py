import glob
import numpy as np
# from PIL import Image
import tensorflow as tf
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob




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
    train_percentage = 0.8
    input_image = np.asarray(input_image)
    print(input_image.shape)
    rand_indices = np.arange(input_image.shape[0])
    np.random.shuffle(rand_indices)
    lenn = int(train_percentage*input_image.shape[0])
    data_train = input_image[rand_indices[0:lenn]]
    data_test = input_image[rand_indices[lenn:]]


    print("Shape of data_train, data_test:", data_train.shape, data_test.shape)
    return data_train, data_test

