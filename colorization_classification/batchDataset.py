"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from scipy import misc
from skimage import color
import _pickle as cPickle

import os
import math
import random
import cv2
#from Queue import Queue
from threading import Thread as Process
from multiprocessing import Process,Queue
import time

from utils import *

from skimage.io import imread
from skimage.transform import resize

class BatchDatset:
    files = []
    images = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, data, image_options={}):
        """
        Intialize a generic file reader with batching for list of files
        :param records_list: list of files to read -
        :param image_options: A dictionary of options for modifying the output image
        Available options:
        resize = True/ False
        resize_size = #size of output image
        color=LAB, RGB, HSV
        """
        print("Initializing Batch Dataset Reader...")
        print(image_options)
        #self.files = records_list
        self.data = data
        self.image_options = image_options
        self._read_images()
        
    #def _read_images(self):
     #   self.images = np.array([self._transform(filename) for filename in self.files])
      #  print (self.images.shape)

    def _read_images(self):
        self.images = []
        self.heatmaps = []
        for i in range(len(self.data)):
            record = self.data[i:i+1]
            image_path = record["image_path"].values[0].replace("/content/","Data_zoo/flowers/")
            raw = record["heatmap"].values[0]
            heatmap = cPickle.loads(bytes(raw, "utf-8"), encoding="bytes")
            heatmap = np.log(heatmap+1)
            heatmap = np.reshape(heatmap, (224,224,1))
            heatmap = np.repeat(heatmap,3, axis=2)
            #print (image_path)
            self.images.append(self._transform(image_path))
            self.heatmaps.append(heatmap)

        self.images = np.array(self.images)
        self.heatmaps = np.array(self.heatmaps)
        print (self.images.shape)
        print (self.heatmaps.shape)

    def _transform(self, filename):
        try:
            image = misc.imread(filename)
            if len(image.shape) < 3:  # make sure images are of shape(h,w,3)
                image = np.array([image for i in range(3)])

            if self.image_options.get("resize", False) and self.image_options["resize"]:
                resize_size = int(self.image_options["resize_size"])
                resize_image = misc.imresize(image,
                                             [resize_size, resize_size])
            else:
                resize_image = image

            #if self.image_options.get("color", False):

             #   option = self.image_options['color']
              #  if option == "LAB":
               #     resize_image = color.rgb2lab(resize_image)
                #elif option == "HSV":
                 #   resize_image = color.rgb2hsv(resize_image)
                #elif option == "RGB":
                 #   pass
        except:
            print ("Error reading file: %s of shape %s" % (filename, str(image.shape)))
            raise

        return np.array(resize_image)

    def get_records(self):
        return np.expand_dims(self.images[:, :, :, 0], axis=3), self.images

    def get_batch_offset(self):
        return self.batch_offset

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        images = self.images[start:end]
        data_l, gt_ab_313, prior_boost_nongray, img_lab = preprocess(images)
        #hm_ab_313 = preprocess_heatmap(heatmaps)
        return data_l, gt_ab_313, prior_boost_nongray, img_lab

    def next_batch_heatmap(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        images = self.images[start:end]
        heatmaps = self.heatmaps[start:end]
        data_l, gt_ab_313, prior_boost_nongray, img_lab = preprocess(images)
        hm_ab_313 = preprocess_heatmap(heatmaps)
        return data_l, gt_ab_313, prior_boost_nongray, img_lab, hm_ab_313

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        images = self.images[indexes]
        heatmaps = self.heatmaps[indexes]
        data_l, gt_ab_313, prior_boost_nongray, img_lab = preprocess(images)
        hm_ab_313 = preprocess_heatmap(heatmaps)
        return data_l, gt_ab_313, prior_boost_nongray, img_lab, hm_ab_313

