
from __future__ import print_function

import sys
import os
import time
import code

import numpy as np
np.random.seed(1234) # for reproducibility?



# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1') 
import lasagne
import theano
import theano.tensor as T


import cPickle as pickle
import gzip

import binary_net

input = T.matrix('inputs')

cnn=lasagne.layers.InputLayer(shape=(None,3),input_var=input)

    
cnn = binary_net.DenseLayer(
            cnn,
            binary=True,
            stochastic=False,
            H=1,
            W_LR_scale=1.0,
            nonlinearity=lasagne.nonlinearities.identity,
            num_units= 2)

cnn = lasagne.layers.BatchNormLayer(
           cnn,
           epsilon=0,
           alpha=0.1)

test_output=lasagne.layers.get_output(cnn,deterministic=True)
train_output=lasagne.layers.get_output(cnn,deterministic=False)
test_loss= T.sum(test_output)
train_loss= T.sum(train_output)

train_fn=theano.function([input],train_output)
test_fn=theano.function([input],test_output)

code.interact(local=locals())
a=np.array([[1,-1,1],[1,-1,1],[1,-1,1]])
b=np.array([[1,1],[1,1],[1,1]])
c=np.array([-10.112,20.344])
d=np.array([6.432,2.22344])
e=np.array([9.42,7.34])
values=lasagne.layers.get_all_param_values(cnn)
values[0]=b
values[1]=c
values[2]=d
values[3]=e
lasagne.layers.set_all_param_values(cnn,values)
lasagne.layers.get_all_param_values(cnn)
test_fn(a)
#train_fn(a)

par=lasagne.layers.get_all_params(cnn)
#grads_test  = T.grad(test_loss,par)
grads_train  = T.grad(train_loss,par[1:4])
gtrain_fn = theano.function([input],grads_train)
gtrain_fn(a)
'''

values=lasagne.layers.get_all_param_values(cnn)
par=lasagne.layers.get_all_params(cnn)
grads_test  = T.grad(test_loss,par)
gtest_fn = theano.function([input],grads_test)

w=np.array([5,5,5,3])
w=w.reshape((1,4))
mdata=[]
mdata.append(w)
lasagne.layers.set_all_param_values(cnn,mdata)
#lasagne.layers.get_all_param_values(cnn)

'''
#x1=scalar('x1',dtype='int32')
#x2=scalar('x2',dtype='int32')
#x3=scalar('x3',dtype='int32')
#x4=scalar('x4',dtype='int32')








