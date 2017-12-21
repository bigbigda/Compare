import tensorflow as tf
import importlib
import tensorflow.python.platform
import os
import numpy as np
from progress.bar import Bar
from datetime import datetime
from tensorflow.python.platform import gfile
from pylearn2.datasets.cifar10 import CIFAR10

import code
from nnUtils import *


x=tf.placeholder(tf.float32,[3,3],name='m_input')
is_training = tf.placeholder(tf.bool,[], name='is_training')

m=BinarizedAffine(2)
y=m(x)
m=BatchNormalization(scale=True,epsilon=0.0,decay=0.9,is_training=is_training)
y=m(y)

loss=tf.reduce_sum(y)
sess=tf.Session()
sess.run(tf.global_variables_initializer())
tvar=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
avar=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

code.interact(local=locals())


gra=tf.gradients(loss,tvar)
a=np.array([[1,-1,1],[1,-1,1],[1,-1,1]])
b=np.array([[1,1],[1,1],[1,1]])
c=np.array([-10,20.344])
d=np.array([6.432,2.22344])
e=np.array([9.42,7.34])

sess.run(tf.assign(avar[0],b))
sess.run(tf.assign(avar[1],c))
sess.run(tf.assign(avar[2],d))
sess.run(tf.assign(avar[3],e))


gres=sess.run(gra,feed_dict={x:a,is_training:False})
res=sess.run(y,feed_dict={x:a,is_training:False})

gtres=sess.run(gra,feed_dict={x:a,is_training:True})
tres=sess.run(y,feed_dict={x:a,is_training:True})
res=sess.run(loss,feed_dict={x:a,is_training:True})

#code.interact(local=locals())
#mydata=sess.run(gra,feed_dict={x:a})


