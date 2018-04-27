import tensorflow as tf 
import numpy as np
from skimage.io import imread
from skimage import color
import sys
import random
from skimage.transform import resize
import read_data as flowers

data_dir = "Data_zoo/flowers"
parent_dir = "/Users/mingjiezhao/Downloads/colorization-tf-master/"
lists_f = open('data/train.txt')
train_images, test_images = flowers.read_dataset(data_dir)
points = np.load('resources/pts_in_hull.npy')
points = points.astype(np.float64)
points = points[None, :, :]
filename_lists = []
probs = np.zeros((313), dtype=np.float64)
num = 0

for img_f in lists_f:
  img_f = img_f.strip()
  filename_lists.append(img_f)
random.shuffle(filename_lists)

#construct graph
in_data = tf.placeholder(tf.float64, [None, 2])
expand_in_data = tf.expand_dims(in_data, axis=1)

distance = tf.reduce_sum(tf.square(expand_in_data - points), axis=2)
index = tf.argmin(distance, axis=1)
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

for i in range(len(train_images)):
  #img_f = img_f.strip()
  #img = imread(img_f)
  record = train_images[i:i+1]
  image_path = record["image_path"].values[0].replace("/content/","Data_zoo/flowers/")
  print (image_path)
  img = imread(parent_dir + image_path)
  img = resize(img, (224, 224), preserve_range=True)
  if len(img.shape)!=3 or img.shape[2]!=3:
    continue
  img_lab = color.rgb2lab(img)
  img_lab = img_lab.reshape((-1, 3))
  img_ab = img_lab[:, 1:]
  nd_index = sess.run(index, feed_dict={in_data: img_ab})
  for i in nd_index:
    i = int(i)
    probs[i] += 1
  print(num)
  sys.stdout.flush()
  num += 1
sess.close()
probs = probs / np.sum(probs)

np.save('probs', probs)
