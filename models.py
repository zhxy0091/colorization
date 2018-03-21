
# coding: utf-8

# In[3]:


import tensorflow as tf
import numpy as np
import math
import tensorflow.contrib.slim as slim


# In[5]:



def lrelu(x, leak=0., name='lrelu'):
    return tf.maximum(leak*x, x)


# In[6]:


def complex_pokemon_model2(X,train=True):
    
  Y = tf.image.convert_image_dtype(X,tf.float32)

  conv_mat = tf.constant(np.array([[0.299,0.587,0.114],[-0.14713,-0.2888,0.436],[0.615,-0.514999,-0.10001]]),dtype = tf.float32)
  inv_conv_mat = tf.constant(np.array([[1,0,1.13983],[1,-0.39465,-0.58060],[1,2.03211,0]]),dtype = tf.float32)

  Y = tf.reshape(Y,[-1,3])

  inputt = tf.matmul(Y,conv_mat)
  inputt = tf.reshape(inputt,[-1,256,256,3])
  Y = tf.reshape(Y,[-1,256,256,3])
  inpp = inputt[:,:,:,1:3]
  inp = inputt[:,:,:,0:1]

  conv1 = lrelu(slim.convolution(inp, 32, 3, stride=1, scope='conv1', normalizer_fn=slim.batch_norm, normalizer_params = {'is_training':train},activation_fn=tf.identity))
  conv2 = lrelu(slim.convolution(conv1, 32, 3, stride=1, scope='conv2', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv3 = lrelu(slim.convolution(conv2, 64, 3, stride=1, scope='conv3', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv4 = lrelu(slim.convolution(conv3, 64, 3, stride=1, scope='conv4', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv5 = lrelu(slim.convolution(conv4, 128, 3, stride=1, scope='conv5', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv6 = lrelu(slim.convolution(conv5, 128, 3, stride=1, scope='conv6', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv7 = lrelu(slim.convolution(conv6, 256, 3, stride=1, scope='conv7', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv8 = lrelu(slim.convolution(conv7, 256, 3, stride=1, scope='conv8', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv9 = lrelu(slim.convolution(conv8, 128, 3, stride=1, scope='conv9', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv10 = lrelu(slim.convolution(conv9, 128, 3, stride=1, scope='conv10', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv11 = lrelu(slim.convolution(conv10, 64, 1, stride=1, scope='conv11', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
  conv12 = lrelu(slim.convolution(conv11, 64, 1, stride=1, scope='conv12', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv13 = lrelu(slim.convolution(conv12, 32, 1, stride=1, scope='conv13', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv14 = lrelu(slim.convolution(conv13, 32, 1, stride=1, scope='conv14', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv15 = lrelu(slim.convolution(conv14, 16, 1, stride=1, scope='conv15', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv16 = lrelu(slim.convolution(conv15, 16, 1, stride=1, scope='conv16', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  conv17 = lrelu(slim.convolution(conv16, 8, 1, stride=1, scope='conv17', normalizer_fn=slim.batch_norm,normalizer_params = {'is_training':train}, activation_fn=tf.identity))
  #if train: conv17 = tf.nn.dropout(conv17, 0.8)
  conv18 = (slim.convolution(conv17, 2, 1, stride=1, scope='conv18', activation_fn=tf.identity))
  #if train: conv18 = tf.nn.dropout(conv18, 0.8)

  a3 = tf.concat((inp,conv18),axis = 3)
  a3 = tf.reshape(a3,[-1,3])
  a2 = tf.matmul(a3,inv_conv_mat)
  a2 = tf.reshape(a2,[-1,256,256,3])

    
  return conv18,a2,inpp,Y
   

