# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
from progress.bar import Bar

import numpy as np
import tensorflow as tf
from data import get_data_provider
from nnUtils import *

def evaluate(model, data,
             batch_size=50,
             checkpoint_dir='./checkpoint',
             device_n = 0):

  with tf.Graph().as_default() as g:

    is_training = tf.placeholder(tf.bool, [], name='is_training')

    x = tf.placeholder(tf.float32, [batch_size, 32, 32, 3])
    yt = tf.placeholder(tf.int32, [batch_size, ])
    yt_onehot = tf.one_hot(yt, depth=10)
    yt_onehot = tf.subtract(tf.multiply(yt_onehot, 2.), 1.)
    device_str = '/gpu:' + str(device_n)
    with tf.device(device_str):
      # Build the Graph that computes the logits predictions
      m = BinarizedWeightOnlySpatialConvolution(128, 3, 3, 1, 1, padding='SAME', bias=True)
      y = m(x)
      m = BatchNormalization(scale=True, epsilon=1e-4, decay=0.9, is_training=is_training)
      y = m(y)
      m = HardTanh()
      y = m(y)
      m = BinarizedSpatialConvolution(128, 3, 3, padding='SAME', bias=True)
      y = m(y)
      m = SpatialMaxPooling(2, 2, 2, 2)
      y = m(y)
      m = BatchNormalization(scale=True, epsilon=1e-4, decay=0.9, is_training=is_training)
      y = m(y)
      m = HardTanh()
      y = m(y)
      m = BinarizedSpatialConvolution(256, 3, 3, padding='SAME', bias=True)
      y = m(y)
      m = BatchNormalization(scale=True, epsilon=1e-4, decay=0.9, is_training=is_training)
      y = m(y)
      m = HardTanh()
      y = m(y)
      m = BinarizedSpatialConvolution(256, 3, 3, padding='SAME', bias=True)
      y = m(y)
      m = SpatialMaxPooling(2, 2, 2, 2)
      y = m(y)
      m = BatchNormalization(scale=True, epsilon=1e-4, decay=0.9, is_training=is_training)
      y = m(y)
      m = HardTanh()
      y = m(y)
      m = BinarizedSpatialConvolution(512, 3, 3, padding='SAME', bias=True)
      y = m(y)
      m = BatchNormalization(scale=True, epsilon=1e-4, decay=0.9, is_training=is_training)
      y = m(y)
      m = HardTanh()
      y = m(y)
      m = BinarizedSpatialConvolution(512, 3, 3, padding='SAME', bias=True)
      y = m(y)
      m = SpatialMaxPooling(2, 2, 2, 2)
      y = m(y)
      m = BatchNormalization(scale=True, epsilon=1e-4, decay=0.9, is_training=is_training)
      y = m(y)
      m = HardTanh()
      y = m(y)
      m = BinarizedAffine2(1024, bias=True)
      y = m(y)
      m = BatchNormalization(scale=True, epsilon=1e-4, decay=0.9, is_training=is_training)
      y = m(y)
      m = HardTanh()
      y = m(y)
      m = BinarizedAffine(1024, bias=True)
      y = m(y)
      m = BatchNormalization(scale=True, epsilon=1e-4, decay=0.9, is_training=is_training)
      y = m(y)
      m = HardTanh()
      y = m(y)
      m = BinarizedAffine(10)
      y = m(y)
      m = BatchNormalization(scale=True, epsilon=1e-4, decay=0.9, is_training=is_training)
      y = m(y)

      # Calculate predictions.
      loss = tf.reduce_mean(tf.maximum(0., 1 - y * yt_onehot) ** 2)
      accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(y, yt, 1), tf.float32))

      # Restore the moving average version of the learned variables for eval.
      # variable_averages = tf.train.ExponentialMovingAverage(
      #    MOVING_AVERAGE_DECAY)
      # variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver()  # variables_to_restore)

    # Configure options for session
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(
      config=tf.ConfigProto(
        log_device_placement=False,
        allow_soft_placement=True,
        gpu_options=gpu_options,
      )
    )

    # 调用最新的保存数据
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir + '/')
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    # 线程
    coord = tf.train.Coordinator()

    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))
      dataset = data[1]
      # num_batches = int(math.ceil(data.size[0] / batch_size))
      num_batches = int(math.floor(dataset.y.shape[0]) / batch_size)
      total_acc = 0  # Counts the number of correct predictions per batch.
      total_loss = 0  # Sum the loss of predictions per batch.
      step = 0
      bar = Bar('Evaluating', max=num_batches,
                suffix='%(percent)d%% eta: %(eta)ds')

      # moving_mean_var = var[]
      while step < num_batches and not coord.should_stop():
        test_x = dataset.X[step * batch_size:(step + 1) * batch_size]
        test_y = dataset.y[step * batch_size:(step + 1) * batch_size]
        acc_val, loss_val = sess.run([accuracy, loss], feed_dict={x: test_x, yt: test_y, is_training: data[3]})
        total_acc += acc_val
        total_loss += loss_val
        step += 1
        bar.next()

      # Compute precision and loss
      total_acc /= num_batches
      total_loss /= num_batches

      bar.finish()

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads)

    return total_acc, total_loss


def main(argv=None):  # pylint: disable=unused-argument
  evaluate()


if __name__ == '__main__':
  FLAGS = tf.app.flags.FLAGS
  tf.app.flags.DEFINE_string('checkpoint_dir', './results/model',
                             """Directory where to read model checkpoints.""")
  tf.app.flags.DEFINE_string('dataset', 'cifar10',
                             """Name of dataset used.""")
  tf.app.flags.DEFINE_string('model_name', 'model',
                             """Name of loaded model.""")

  FLAGS.log_dir = FLAGS.checkpoint_dir + '/log/'

  tf.app.run()
