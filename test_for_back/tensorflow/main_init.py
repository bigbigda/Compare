import tensorflow as tf
import importlib
import tensorflow.python.platform
import os
import numpy as np
from progress.bar import Bar
from datetime import datetime
from tensorflow.python.platform import gfile
from data import *
from evaluate import evaluate
from pylearn2.datasets.cifar10 import CIFAR10

from nnUtils import *
import code

timestr = '-'.join(str(x)
                   for x in list(tuple(datetime.now().timetuple())[1:5]))
MOVING_AVERAGE_DECAY = 0.997
FLAGS = tf.app.flags.FLAGS
savegap = 10
# learning rate schedule


# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 50,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_epochs', 500,
                            """Number of epochs to train. -1 for unlimited""")
tf.app.flags.DEFINE_integer('learning_rate', 1e-3,
                            """Initial learning rate used.""")
tf.app.flags.DEFINE_string('model', 'BNN_cifar10',
                           """Name of loaded model.""")
tf.app.flags.DEFINE_string('save', timestr,
                           """Name of saved dir.""")
tf.app.flags.DEFINE_string('load', None,
                           """Name of loaded dir.""")
tf.app.flags.DEFINE_string('dataset', 'cifar10',
                           """Name of dataset used.""")
tf.app.flags.DEFINE_string('gpu', True,
                           """use gpu.""")
tf.app.flags.DEFINE_string('device', 0,
                           """which gpu to use.""")
tf.app.flags.DEFINE_string('summary', True,
                           """Record summary.""")
tf.app.flags.DEFINE_string('log', 'ERROR',
                           'The threshold for what messages will be logged '
                           """DEBUG, INFO, WARN, ERROR, or FATAL.""")

FLAGS.checkpoint_dir = './results/' + FLAGS.save
FLAGS.log_dir = FLAGS.checkpoint_dir
lr_start = 0.001
lr_end = 0.0000003
lr_decay = (lr_end / lr_start)**(1. / FLAGS.num_epochs)


def count_params(var_list):
    num = 0
    for var in var_list:
        if var is not None:
            num += var.get_shape().num_elements()
    return num


def add_summaries(scalar_list=[], activation_list=[], var_list=[], grad_list=[]):

    for var in scalar_list:
        if var is not None:
            tf.summary.scalar(var.op.name, var)

    for grad, var in grad_list:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)
    for var in var_list:
        if var is not None:
            tf.summary.histogram(var.op.name, var)
            sz = var.get_shape().as_list()
            # 可视化第一层的卷积核
            # if len(sz) == 4 and sz[2] == 3:
            #    kernels = tf.transpose(var, [3, 0, 1, 2])
            #    tf.summary.image(var.op.name + '/kernels',
            #                    group_batch_images(kernels), max_outputs=1)

    for activation in activation_list:
        if activation is not None:
            tf.summary.histogram(activation.op.name +
                                 '/activations', activation)

# learning_rate * decay_rate ^ (global_step/decay_steps)


def _learning_rate_decay_fn(learning_rate, global_step):
    return tf.train.exponential_decay(
        learning_rate,
        global_step,
        decay_steps=1000,
        decay_rate=0.9,
        staircase=True)


learning_rate_decay_fn = _learning_rate_decay_fn


def shuffle(X, y, shuffle_parts=1):

    chunk_size = int(len(X) / shuffle_parts)
    shuffled_range = list(range(chunk_size))

    X_buffer = np.copy(X[0:chunk_size])
    y_buffer = np.copy(y[0:chunk_size])

    for k in range(shuffle_parts):  # 对每一个区块进行随机排序

        np.random.shuffle(shuffled_range)

        for i in range(chunk_size):
            X_buffer[i] = X[k * chunk_size + shuffled_range[i]]
            y_buffer[i] = y[k * chunk_size + shuffled_range[i]]

        X[k * chunk_size:(k + 1) * chunk_size] = X_buffer
        y[k * chunk_size:(k + 1) * chunk_size] = y_buffer

    return X, y


def train(model, dataset,
          batch_size=FLAGS.batch_size,
          learning_rate=FLAGS.learning_rate,
          log_dir='./log',
          checkpoint_dir='./checkpoint',
          num_epochs=-1):

    x = tf.placeholder(tf.float32, [batch_size, 32, 32, 3], name='image-input')
    yt = tf.placeholder(tf.int32, [batch_size, ], name='y-label')
    is_training = tf.placeholder(tf.bool,[], name='is_training')
    global_step = tf.get_variable('global_step', shape=[], dtype=tf.int64,
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

    # yt_ for squared hinge loss
    yt_onehot = tf.one_hot(yt, depth=10)
    yt_onehot = tf.subtract(tf.multiply(yt_onehot, 2.), 1.)

    if 0:
        device_str = '/gpu:' + str(FLAGS.device)
    else:
        device_str = '/cpu:0'
    with tf.device(device_str):

        m=BinarizedWeightOnlySpatialConvolution(128, 3, 3, 1, 1, padding='SAME', bias=True)
        y=m(x)
        m=BatchNormalization(scale=True, epsilon=1e-4, decay=0.9, is_training=is_training)
        y=m(y)
        m=HardTanh()
        y=m(y)
        m =BinarizedSpatialConvolution(128, 3, 3, padding='SAME', bias=True)
        y=m(y)
        m =SpatialMaxPooling(2, 2, 2, 2)
        y=m(y)
        m =BatchNormalization(scale=True, epsilon=1e-4, decay=0.9, is_training=is_training)
        y=m(y)
        m =HardTanh()
        y=m(y)
        m =BinarizedSpatialConvolution(256, 3, 3, padding='SAME', bias=True)
        y=m(y)
        m =BatchNormalization(scale=True, epsilon=1e-4, decay=0.9, is_training=is_training)
        y=m(y)
        m =HardTanh()
        y=m(y)
        m =BinarizedSpatialConvolution(256, 3, 3, padding='SAME', bias=True)
        y=m(y)
        m =SpatialMaxPooling(2, 2, 2, 2)
        y=m(y)
        m =BatchNormalization(scale=True, epsilon=1e-4, decay=0.9, is_training=is_training)
        y=m(y)
        m =HardTanh()
        y=m(y)
        m =BinarizedSpatialConvolution(512, 3, 3, padding='SAME', bias=True)
        y=m(y)
        m =BatchNormalization(scale=True, epsilon=1e-4, decay=0.9, is_training=is_training)
        y=m(y)
        m =HardTanh()
        y=m(y)
        m =BinarizedSpatialConvolution(512, 3, 3, padding='SAME', bias=True)
        y=m(y)
        m =BatchNormalization(scale=True, epsilon=1e-4, decay=0.9, is_training=is_training)
        y=m(y)
        '''
        m =HardTanh()
        y=m(y)
        m =BinarizedAffine2(1024, bias=True)
        y=m(y)
        m =BatchNormalization(scale=True, epsilon=1e-4, decay=0.9, is_training=is_training)
        y=m(y)
        m =HardTanh()
        y=m(y)
        m =BinarizedAffine(1024, bias=True)
        y=m(y)
        m =BatchNormalization(scale=True, epsilon=1e-4, decay=0.9, is_training=is_training)
        y=m(y)
        m =HardTanh()
        y=m(y)
        m =BinarizedAffine(10)
        y=m(y)
        m =BatchNormalization(scale=True, epsilon=1e-4, decay=0.9, is_training=is_training)
        y=m(y)
        '''

        # y = model(x)
        # all_var = tf.get_collection(tf.GraphKeys.VARIABLES)[1:]
        # weight_bias = []
        # for i in range(9):
        #     weight_bias.append(all_var[6 * i])
        #     weight_bias.append(all_var[6 * i + 1])
        # import code
        # code.interact(locals())
        # Define loss and optimizer
        with tf.name_scope('objective'):
            loss = tf.reduce_mean(tf.maximum(0., 1 - y) ** 2)

            accuracy = tf.reduce_mean(
                tf.cast(y, tf.float32), name="accuracy")  # y[10] yt[1]

        #tf.summary.scalar('loss', loss)
        #tf.summary.scalar('accuracy', accuracy)
        npzfile = np.load('gb2.npz')
        data = []
        for i in range(54):
            name = "arr_" + str(i)
            data = data + [npzfile[name]]

        data[36]=np.concatenate((data[36],data[36],data[36],data[36]))
        for i in range(6):
            data[i * 6] = np.rot90(data[i * 6], axes=(2, 3))
            data[i * 6] = np.rot90(data[i * 6], axes=(2, 3))
            data[i * 6] = np.transpose(data[i * 6], (2, 3, 1, 0))
            data[i * 6 + 5] = np.square(np.divide(1., data[i * 6 + 5])) - (1e-4)

        #data[19]=data[19]*1e7
        #data[7]=data[7]*1e5

        data[6 * 6 + 5] = np.square(np.divide(1., data[6 * 6 + 5])) - (1e-4)
        data[7 * 6 + 5] = np.square(np.divide(1., data[7 * 6 + 5])) - (1e-4)
        data[8 * 6 + 5] = np.square(np.divide(1., data[8 * 6 + 5])) - (1e-4)

        LR_scale_1 = tf.constant(28.0357, tf.float32, [
                                 3, 3, 3, 128], name='BinarizedWeightOnlySpatialConvolution_weight')
        LR_scale_2 = tf.constant(39.1918, tf.float32, [
                                 3, 3, 128, 128], name='BinarizedSpatialConvolution_weight')
        LR_scale_3 = tf.constant(
            48, tf.float32, [3, 3, 128, 256], name='BinarizedSpatialConvolution_1_weight')
        LR_scale_4 = tf.constant(55.4256, tf.float32, [
                                 3, 3, 256, 256], name='BinarizedSpatialConvolution_2_weight')
        LR_scale_5 = tf.constant(67.8822, tf.float32, [
                                 3, 3, 256, 512], name='BinarizedSpatialConvolution_3weight')
        LR_scale_6 = tf.constant(78.3837, tf.float32, [
                                 3, 3, 512, 512], name='BinarizedSpatialConvolution_4weight')
        LR_scale_7 = tf.constant(78.3837, tf.float32, [
                                 32768, 1024], name='Affine_weight')
        LR_scale_8 = tf.constant(36.9504, tf.float32, [
                                 1024, 1024], name='Affine_1_weight')
        LR_scale_9 = tf.constant(26.2552, tf.float32, [
                                 1024, 10], name='Affine_2_weight')
        LR_scale = [LR_scale_1, LR_scale_2, LR_scale_3, LR_scale_4,
                    LR_scale_5, LR_scale_6, LR_scale_7, LR_scale_8, LR_scale_9]

        learning_rate_ = tf.train.exponential_decay(learning_rate, global_step=global_step,
                                                    decay_steps=
                                                        1000*500,
                                                    decay_rate=lr_decay, staircase=True)
        # 默认参数 beta1 = 0.9 beta2 = 0.999 epsilon = 1e-8
        optimizer = tf.train.AdamOptimizer(10)
        #optimizer = tf.train.AdamOptimizer(learning_rate_)
        grads_and_vars = optimizer.compute_gradients(
            loss, tf.trainable_variables())
        var_clip_list = tf.get_collection('weights')
        # import code
        # code.interact(local=locals())
        grads_and_vars_update = []
        idx = 0
        for grad, var in grads_and_vars:
            if grad is not None and var in var_clip_list:
                LR_scale_val = LR_scale[idx]
                grad = tf.multiply(grad, LR_scale_val)
                idx = idx + 1
            grads_and_vars_update.append((grad, var))
        gradients_opt = optimizer.apply_gradients(
            [grads_and_vars_update[1]], global_step)

        clip_value_ = []
        for var in var_clip_list:
            #clip_value = tf.clip_by_value(var, -1.0, +1.0)  # Tensor
            clip_value=tf.assign(var,tf.clip_by_value(var,-1.0,+1.0))
            clip_value_.append(clip_value)
            train_op = tf.group(*clip_value_)
        all_var = tf.get_collection(tf.GraphKeys.VARIABLES)[1:]
        update_var = [None]*54
        for i in range(6):
            update_var[i*6] = all_var[i*6]
            update_var[i*6+1] = all_var[i*6+1]
            if (i<6):
              update_var[i*6+2] = all_var[i*6+3]
              update_var[i*6+3] = all_var[i*6+2]
            else:
              update_var[i*6+2] = all_var[i*6+2]
              update_var[i*6+3] = all_var[i*6+3]
            update_var[i*6+4] = all_var[i*6+4]
            update_var[i*6+5] = all_var[i*6+5]
        assign = []
        for i in range(36):
            opt = tf.assign(update_var[i],data[i],validate_shape=True)
            assign.append(opt)
        assign_op = tf.group(*assign)


        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.InteractiveSession(
        config=tf.ConfigProto(
            log_device_placement=False,
            allow_soft_placement=True,
            gpu_options=gpu_options,
        )
    )



        sess.run(tf.global_variables_initializer())

        num_batches = len(dataset[0].y) / batch_size
        epoch = 0
        train_set, test_set = dataset[0], dataset[1]

        def evaluate():
           test_loss = 0.0
           test_accu = 0.0
           for i in range(20):
               loss_, acc_= sess.run([loss, accuracy],feed_dict={
                     x: test_set.X[50*i:50*(i+1)], yt: test_set.y[50*i:50*(i+1)],is_training:False})
               test_loss += loss_
               test_accu += acc_
           test_loss = test_loss/20
           test_accu = test_accu/20
           print ("test loss: ", test_loss)
           print ("test accu: ", test_accu)

        sess.run(assign_op)
        evaluate()
        train_X, train_Y  = train_set.X ,train_set.y
        code.interact(local=locals())           
        # import tensorflow as tf
        # import numpy as np
        # gra= tf.gradients(loss,all_var[18:22])
        # mydata=sess.run(gra,feed_dict={x:train_X[0:50],yt:train_Y[0:50],is_training:False})

        while epoch != 3:
            epoch += 1
            curr_step = 0

            print('Started epoch %d' % epoch)
            # train_X, train_Y = shuffle(train_set.X, train_set.y)
            train_X, train_Y  = train_set.X ,train_set.y
            acc_value = 0.
            loss_value = 0.
            num_batches = 0
            mini_num = 0
            all_var = tf.get_collection(tf.GraphKeys.VARIABLES)[1:]
            mypar=all_var[2]
            print (sess.run(mypar))
            while (mini_num<1):
                train_x = train_X[0:50]
                train_yt = train_Y[0:50]

                mini_num += 1
                #_,  loss_, acc_,res = sess.run([update_ops, loss, accuracy,y], feed_dict={
                _, _, loss_, acc_,res = sess.run([update_ops,gradients_opt, loss, accuracy,y], feed_dict={
                     x: train_x, yt: train_yt, is_training: True})
                _ = sess.run([train_op])
                
                #loss_, acc_= sess.run([loss, accuracy],feed_dict={
                #               x: train_x, yt: train_yt,is_training:dataset[3]})
                acc_value += acc_
                loss_value += loss_
                num_batches += 1
                #print (res)
            all_var = tf.get_collection(tf.GraphKeys.VARIABLES)[1:]
            code.interact(local=locals())     
            mypar=all_var[2]
            print (sess.run(mypar))
            acc_value = acc_value / float(num_batches)
            loss_value = loss_value / float(num_batches)

            # 保存当前训练数据用于模型评估

            print('Finished epoch %d' % epoch)
            print('batches:%d' % num_batches)
            # print('Learning rate %f' % LR)
            print('Training Accuracy: %.3f' % acc_value)
            print('Training Loss: %.3f' % loss_value)

            # 验证
            evaluate()


def main(argv=None):  # pylint: disable=unused-argument
    if not gfile.Exists(FLAGS.checkpoint_dir):
        # gfile.DeleteRecursively(FLAGS.checkpoint_dir)
        gfile.MakeDirs(FLAGS.checkpoint_dir)
        model_file = os.path.join('models', FLAGS.model + '.py')
        assert gfile.Exists(model_file), 'no model file named: ' + model_file
        gfile.Copy(model_file, FLAGS.checkpoint_dir + '/model.py')
    print(FLAGS.model)
    m = importlib.import_module('models.' + FLAGS.model, 'models')
    # data = get_data_provider(FLAGS.dataset, training=True)
    print('Loading CIFAR-10 dataset...')

    train_set = CIFAR10(which_set="train", start=0, stop=45000)
    test_set = CIFAR10(which_set="test")


    # bc01 format
    # Inputs in the range [-1,+1]
    # [0-255]-[0-1]-[0-2]-[-1,1]
    train_set.X = np.reshape(np.subtract(np.multiply(2. / 255., train_set.X), 1.), (-1, 3, 32, 32))
    test_set.X = np.reshape(np.subtract(np.multiply(2. / 255., test_set.X), 1.), (-1, 3, 32, 32))
    # 32*32*3
    train_set.X = train_set.X.transpose([0, 2, 3, 1])
    test_set.X = test_set.X.transpose([0, 2, 3, 1])
    # flatten targets
    train_set.y = np.hstack(train_set.y)
    test_set.y = np.hstack(test_set.y)
    train_training = True
    test_training = False
    # 训练集和测试集数据
    data = [train_set, test_set, train_training, test_training]
    # m.model 是一个函数
    train(m.model, data,
          batch_size=FLAGS.batch_size,
          checkpoint_dir=FLAGS.checkpoint_dir,
          log_dir=FLAGS.log_dir,
          num_epochs=FLAGS.num_epochs)


if __name__ == '__main__':
    tf.app.run()
