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


x=tf.placeholder(tf.float32,[2,2],name='m_input')
#v1=tf.get_variable("v1",1)
#v2=tf.get_variable("v2",2)
#v3=tf.get_variable("v3",3)
#v4=tf.get_variable("v4",4)
vt=tf.get_variable("v4",shape=(2,2))

#v2d=[[x[0,0]+v1,x[0,1]+v2],[x[1,0]+v3,x[1,1]+v4]]
v2d=vt+x
code.interact(local=locals())
v4d=tf.reshape(v2d,[1,2,2,1])
m =SpatialMaxPooling(2, 2, 2, 2)

y=m(v4d)

loss=tf.sum(y)

sess=tf.Session()

sess.run(tf.global_variables_initializer())

trainable_var=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
gra=tf.gradients(loss,trainable_var)
a=np.array([[0,0],[0,0]])

code.interact(local=locals())
mydata=sess.run(gra,feed_dict={x:a})


