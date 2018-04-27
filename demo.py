
# coding: utf-8

# In[1]:



import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import TensorflowUtils as utils
import read_data as flowers

import datetime
import batchReader as dataset
from six.moves import xrange
import os
import glob
from scipy import misc
from skimage import color
import skimage
import skimage.io
import skimage.transform

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "16", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "demo/", "path to dataset")

tf.flags.DEFINE_string("color_model_dir", "logs/model_heatmap/", "path to model")
#tf.flags.DEFINE_string("color_model_dir", "logs/model_no_heatmap/", "path to model")

tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_float("beta1", "0.9", "Beta 1 value to use in Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test")
tf.flags.DEFINE_bool('restore_model', "False", "Restore Model: True/ False")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

MAX_ITERATION = int(1e5 + 1)
IMAGE_SIZE = 256
ADVERSARIAL_LOSS_WEIGHT = 1e-3

def vgg_net(weights, image):
    layers = (
        # 'conv1_1', 'relu1_1',
        'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i + 2][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if FLAGS.debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

    return net


def generator(images, train_phase):
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    weights = np.squeeze(model_data['layers'])

    with tf.variable_scope("generator") as scope:
        W0 = utils.weight_variable([3, 3, 1, 64], name="W0")
        b0 = utils.bias_variable([64], name="b0")
        conv0 = utils.conv2d_basic(images, W0, b0)
        hrelu0 = tf.nn.relu(conv0, name="relu")

        image_net = vgg_net(weights, hrelu0)
        vgg_final_layer = image_net["relu5_3"]

        pool5 = utils.max_pool_2x2(vgg_final_layer)

        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, pool5.get_shape()[3].value], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(pool5, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        shape = tf.shape(images)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], 2])
        W_t3 = utils.weight_variable([16, 16, 2, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([2], name="b_t3")
        pred = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

    return tf.concat(axis=3, values=[images, pred], name="pred_image")


def train(loss, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1)
    grads = optimizer.compute_gradients(loss, var_list=var_list)
    for grad, var in grads:
        utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


def transform(dir):
    images = []
    gray_images = []
    for filename in glob.glob("demo/*.jpg"):
        print(filename)
        image = skimage.io.imread(filename)
        resize_image = skimage.transform.resize( image, [224, 224] , mode='constant')
        
        if(len(resize_image.shape)==3):
            images.append(resize_image)
            resize_image = color.rgb2grey(resize_image)
        else:
            tmp = np.reshape(resize_image, (224,224,1))
            images.append(np.repeat(tmp,3, axis=2))
        gray_image = resize_image*100
        gray_image = gray_image.reshape((224,224,1))
        gray_images.append(np.array(gray_image))
    
    images = np.array(images)
    gray_images = np.array(gray_images)
    return gray_images, images


# In[8]:


images = []
gray_images = []
for filename in glob.glob("demo/*.jpg"):
    print(filename)
    image = skimage.io.imread(filename)
    resize_image = skimage.transform.resize( image, [224, 224] , mode='constant')

    if(len(resize_image.shape)==3):
        images.append(resize_image)
        resize_image = color.rgb2grey(resize_image)
    else:
        tmp = np.reshape(resize_image, (224,224,1))
        images.append(np.repeat(tmp,3, axis=2))
        
    gray_image = resize_image*100.0
    gray_image = gray_image.reshape((224,224,1))
    gray_images.append(np.array(gray_image))

images = np.array(images)
gray_images = np.array(gray_images)


# In[2]:


def main(argv=None):
    print("Setting up network...")
    train_phase = tf.placeholder(tf.bool, name="train_phase")
    images = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='L_image')
    lab_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="LAB_image")

    pred_image = generator(images, train_phase)

    gen_loss_mse = tf.reduce_mean(2 * tf.nn.l2_loss(pred_image - lab_images)) / (IMAGE_SIZE * IMAGE_SIZE * 100 * 100)
    tf.summary.scalar("Generator_loss_MSE", gen_loss_mse)

    train_variables = tf.trainable_variables()
    for v in train_variables:
        utils.add_to_regularization_and_summary(var=v)

    train_op = train(gen_loss_mse, train_variables)
    
    print("Setting up session")
    sess = tf.Session()
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)
    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(FLAGS.color_model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    else:
        print("Model is not found")
        return
    
    print("Reading demo image ...")
    l_image, color_images = transform(FLAGS.data_dir)
    
    print(l_image.shape)
    print(color_images.shape)
    feed_dict = {images: l_image, lab_images: color_images, train_phase: False}
    save_dir = FLAGS.data_dir
    pred = sess.run(pred_image, feed_dict=feed_dict)
        
    for itr in range(color_images.shape[0]):
        #utils.save_image(color_images[idx], save_dir, "gt" + str(itr))
        if FLAGS.color_model_dir=='logs/model_heatmap/':
            utils.save_image(pred[itr].astype(np.float64), save_dir, "heat" + str(itr))
        else:
            utils.save_image(pred[itr].astype(np.float64), save_dir, "pred" + str(itr))
    print("--- Images saved on demo run ---")
   
 


    
    
    


# In[3]:


if __name__ == "__main__":
    tf.app.run()

