import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os, sys, inspect
import random
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
import glob

utils_path = os.path.abspath(
    os.path.realpath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..")))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
import TensorflowUtils as utils


DATA_URL = 'http://data.csail.mit.edu/places/places365/val_256.tar'

def read_dataset(data_dir, testing_percentage=0.0, validation_percentage=0.2):
    # input_image = []
    # i = 0
    # for filename in glob.iglob(data_dir+'/*',recursive=True):
    #     #print("file_name", filename) 
    #     img = mpimg.imread(filename)
    #     if(img.shape == (256,256,3)):
    #         input_image.append(filename)
    #     else:
    #         print("file_name", filename, "is invalid") 
    #     if(i%10000 == 0):
    #         print("loading", i)
    #     if(i >1000):
    #         break

    #     i+=1
    
    # np.random.shuffle(input_image)
    # no_of_images = len(input_image)
    # training_images = input_image

    # validation_offset = int(validation_percentage * no_of_images)
    # validation_images = training_images[:validation_offset]
    # test_offset = int(testing_percentage * no_of_images)
    # testing_images = training_images[validation_offset:validation_offset + test_offset]
    # training_images = training_images[validation_offset + test_offset:]

    # print ("Training: %d, Validation: %d, Test: %d" % (
    #     len(training_images), len(validation_images), len(testing_images)))
    # return training_images, testing_images, validation_images

    pickle_filename = "places.pickle"
    pickle_filepath = os.path.join(data_dir, pickle_filename)
    if not os.path.exists(pickle_filepath):
        utils.maybe_download_and_extract(data_dir, DATA_URL)
        places_folder = (DATA_URL.split("/")[-1]).split(os.path.extsep)[0]
        result = create_image_lists(os.path.join(data_dir, places_folder), testing_percentage, validation_percentage)
        print ("Pickling ...")
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    else:
        print ("Found pickle file!")

    with open(pickle_filepath, 'rb') as f:
        result = pickle.load(f)
        training_images = result['train']
        testing_images = result['test']
        validation_images = result['validation']
        del result
    print ("Training: %d, Validation: %d, Test: %d" % (len(training_images), len(validation_images), len(testing_images)))
    return training_images, testing_images, validation_images


def create_image_lists(image_dir, testing_percentage=0.0, validation_percentage=0.2):
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    image_list = []

    file_list = []
    file_glob = image_dir+'/*.jpg'
    #file_list.extend(glob.glob(file_glob))

    i=0
    for filename in glob.glob(file_glob):
        img = mpimg.imread(filename)
        if(img.shape == (256,256,3)):
            file_list.append(filename)
        else:
            print("file_name", filename, "is invalid") 
        if(i%10000 == 0):
            print("loading", i)
        # if(i >1000):
        #     break
        i += 1

    if not file_list:
        print('No files found')
    else:
        image_list = file_list

    random.shuffle(image_list)
    no_of_images = len(image_list)
    print ('No. of Image files: %d' % no_of_images)

    training_images = image_list
    validation_offset = int(validation_percentage * no_of_images)
    validation_images = training_images[:validation_offset]
    test_offset = int(testing_percentage * no_of_images)
    testing_images = training_images[validation_offset:validation_offset + test_offset]
    training_images = training_images[validation_offset + test_offset:]

    result = {
        'train': training_images,
        'test': testing_images,
        'validation': validation_images,
    }
    return result


