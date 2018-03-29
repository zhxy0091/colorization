import glob
import numpy as np
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob

def read_dataset(data_dir, testing_percentage=0.0, validation_percentage=0.2):
    input_image = []
    i = 0
    for filename in glob.iglob(data_dir+'/*',recursive=True):
        #print("file_name", filename) 
        img = mpimg.imread(filename)
        if(img.shape == (256,256,3)):
            input_image.append(filename)
        else:
            print("file_name", filename, "is invalid") 
        if(i%10000 == 0):
            print("loading", i)
        if(i >1000):
            break

        i+=1
    
    np.random.shuffle(input_image)
    no_of_images = len(input_image)
    training_images = input_image

    validation_offset = int(validation_percentage * no_of_images)
    validation_images = training_images[:validation_offset]
    test_offset = int(testing_percentage * no_of_images)
    testing_images = training_images[validation_offset:validation_offset + test_offset]
    training_images = training_images[validation_offset + test_offset:]

    print ("Training: %d, Validation: %d, Test: %d" % (
        len(training_images), len(validation_images), len(testing_images)))
    return training_images, testing_images, validation_images


