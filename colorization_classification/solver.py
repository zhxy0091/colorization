from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from utils import *
import tensorflow as tf
import numpy as np
import re
import cv2
from ops import *
from net import Net
from data import DataSet
import time
from datetime import datetime
import os
import sys
import batchDataset as dataset
import read_data as flowers

#FLAGS = tf.flags.FLAGS
#tf.flags.DEFINE_integer("batch_size", "16", "batch size for training")
#tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
#tf.flags.DEFINE_string("data_dir", "Data_zoo/flowers/", "path to dataset")
#tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
#tf.flags.DEFINE_float("beta1", "0.9", "Beta 1 value to use in Adam Optimizer")
#tf.flags.DEFINE_string("model_dir", "models/", "Path to vgg model mat")
##tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
#tf.flags.DEFINE_string('mode', "train", "Mode train/ test")
#tf.flags.DEFINE_bool('restore_model', "True", "Restore Model: True/ False")

def save_image(image, save_dir, name):
  """
  Save image by unprocessing and converting to rgb.
  :param image: iamge to save
  :param save_dir: location to save image at
  :param name: prefix to save filename
  :return:
  """
  image = color.lab2rgb(image)
  #img_rgb = decode(data_l, conv8_313,2.63)
  scipy.misc.imsave(os.path.join(save_dir, name + ".jpg"), image)
class Solver(object):

  def __init__(self, train=True, common_params=None, solver_params=None, net_params=None, dataset_params=None):
    if common_params:
      self.device_id = int(common_params['gpus'])
      self.image_size = int(common_params['image_size'])
      self.height = self.image_size
      self.width = self.image_size
      self.batch_size = int(common_params['batch_size'])
      self.num_gpus = 1
      self.restore_model = True
      self.logs_dir = "logs/"
      self.data_dir = "Data_zoo/flowers"
    if solver_params:
      self.learning_rate = float(solver_params['learning_rate'])
      self.moment = float(solver_params['moment'])
      self.max_steps = int(solver_params['max_iterators'])
      self.train_dir = str(solver_params['train_dir'])
      self.lr_decay = float(solver_params['lr_decay'])
      self.decay_steps = int(solver_params['decay_steps'])
    self.train = train
    self.net = Net(train=train, common_params=common_params, net_params=net_params)
    #self.dataset = DataSet(common_params=common_params, dataset_params=dataset_params)
    self.train_images, self.test_images = flowers.read_dataset(self.data_dir)
    image_options = {"resize": True, "resize_size": 224, "color": "LAB"}
    self.batch_reader = dataset.BatchDatset(self.train_images, image_options)
    self.batch_reader_test = dataset.BatchDatset(self.test_images, image_options)
  def construct_graph(self, scope):
    with tf.device('/gpu:' + str(self.device_id)):
      self.data_l = tf.placeholder(tf.float32, (self.batch_size, self.height, self.width, 1))
      self.gt_ab_313 = tf.placeholder(tf.float32, (self.batch_size, int(self.height / 4), int(self.width / 4), 313))
      self.prior_boost_nongray = tf.placeholder(tf.float32, (self.batch_size, int(self.height / 4), int(self.width / 4), 1))

      self.conv8_313 = self.net.inference(self.data_l)
      new_loss, g_loss = self.net.loss(scope, self.conv8_313, self.prior_boost_nongray, self.gt_ab_313)
      tf.summary.scalar('new_loss', new_loss)
      tf.summary.scalar('total_loss', g_loss)
    return new_loss, g_loss, self.conv8_313

  def pred_image(self, images):

    #img = img[None, :, :, None]
    #data_l = (img.astype(dtype=np.float32)) / 255.0 * 100 - 50

    #data_l = tf.placeholder(tf.float32, shape=(None, None, None, 1))
    autocolor = Net(train=False)

    conv8_313 = autocolor.inference(images)
    
    return conv8_313

    #imsave('color.jpg', img_rgb)
  
  def train_model(self):
    
    with tf.device('/gpu:' + str(self.device_id)):
      tf.reset_default_graph() 
      self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
      learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                           self.decay_steps, self.lr_decay, staircase=True)
      opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta2=0.99)
      images = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='L_image')
      with tf.name_scope('gpu') as scope:
        self.data_l = tf.placeholder(tf.float32, (self.batch_size, self.height, self.width, 1))
        self.gt_ab_313 = tf.placeholder(tf.float32, (self.batch_size, int(self.height / 4), int(self.width / 4), 313))
        self.prior_boost_nongray = tf.placeholder(tf.float32, (self.batch_size, int(self.height / 4), int(self.width / 4), 1))

        conv8_313 = self.net.inference(self.data_l)
        new_loss, g_loss = self.net.loss(scope, conv8_313, self.prior_boost_nongray, self.gt_ab_313)
        tf.summary.scalar('new_loss', new_loss)
        tf.summary.scalar('total_loss', g_loss)
        self.total_loss = g_loss
        self.summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
      grads = opt.compute_gradients(new_loss)

      self.summaries.append(tf.summary.scalar('learning_rate', learning_rate))

      for grad, var in grads:
        if grad is not None:
          self.summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

      apply_gradient_op = opt.apply_gradients(grads, global_step=self.global_step)

      for var in tf.trainable_variables():
        self.summaries.append(tf.summary.histogram(var.op.name, var))

      variable_averages = tf.train.ExponentialMovingAverage(
          0.999, self.global_step)
      variables_averages_op = variable_averages.apply(tf.trainable_variables())

      train_op = tf.group(apply_gradient_op, variables_averages_op, conv8_313)

      saver = tf.train.Saver(write_version=1)
      summary_op = tf.summary.merge(self.summaries)
      init =  tf.global_variables_initializer()
      config = tf.ConfigProto(allow_soft_placement=True)
      config.gpu_options.allow_growth = True
      sess = tf.Session(config=config)
      sess.run(init)
      #saver1.restore(sess, './models/model.ckpt')
      #nilboy
      
      if self.restore_model:
        ckpt = tf.train.get_checkpoint_state(self.logs_dir + 'model_without_heatmap/')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored...")
      summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph)
      for step in range(self.max_steps):
        print(step)
        start_time = time.time()
        t1 = time.time()
        data_l, gt_ab_313, prior_boost_nongray,color_image = self.batch_reader.next_batch(16)
        t2 = time.time()
        _, loss_value = sess.run([train_op, self.total_loss], feed_dict={self.data_l:data_l, self.gt_ab_313:gt_ab_313, self.prior_boost_nongray:prior_boost_nongray})
        duration = time.time() - start_time
        t3 = time.time()
        print('io: ' + str(t2 - t1) + '; compute: ' + str(t3 - t2))
        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step % 1 == 0:
          num_examples_per_step = self.batch_size * self.num_gpus
          examples_per_sec = num_examples_per_step / duration
          sec_per_batch = duration / self.num_gpus

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), step, loss_value,
                               examples_per_sec, sec_per_batch))
        
        if step % 10 == 0:
          summary_str = sess.run(summary_op, feed_dict={self.data_l:data_l, self.gt_ab_313:gt_ab_313, self.prior_boost_nongray:prior_boost_nongray})
          summary_writer.add_summary(summary_str, step)

        # Save the model checkpoint periodically.
        if step % 100 == 0:
          checkpoint_path = os.path.join(self.train_dir, 'model.ckpt')
          saver.save(sess, self.logs_dir + "model.ckpt", global_step=step)
          pred = sess.run(conv8_313, feed_dict={self.data_l:data_l, self.gt_ab_313:gt_ab_313, self.prior_boost_nongray:prior_boost_nongray})
          idx = np.random.randint(0, self.batch_size)
          save_dir = os.path.join(self.logs_dir, "image_checkpoints")
          save_image(color_image[idx], save_dir, "gt" + str(step // 1))
          print (pred[idx].shape)
          #pred[idx] = tf.expand_dims(pred[idx], 0)
          #print (pred[idx].shape)
          predict = decode(data_l[idx:idx+1], pred[idx:idx+1],2.63)

          save_image(predict.astype(np.float64), save_dir, "pred" + str(step // 1))
          print("%s --> Model saved" % datetime.now())

        if step % 1000 == 0:
          count = 16
          data_l_test, gt_ab_313_test, prior_boost_nongray_test, color_image_test = self.batch_reader_test.get_random_batch(count)
          feed_dict = {self.data_l:data_l_test, self.gt_ab_313:gt_ab_313_test, self.prior_boost_nongray:prior_boost_nongray_test}
          save_dir = os.path.join(self.logs_dir, "image_pred")
          pred = sess.run(conv8_313, feed_dict=feed_dict)
          for itr in range(count):
            save_image(color_image_test[itr], save_dir, "gt" + str(itr))
            predict = decode(data_l_test[itr:itr+1], pred[itr:itr+1],2.63)
            save_image(predict, save_dir, "pred" + str(itr))
          print("--- Images saved on test run ---")