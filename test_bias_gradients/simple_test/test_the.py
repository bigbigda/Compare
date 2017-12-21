
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


input = T.matrix('inputs')

cnn=lasagne.layers.InputLayer(shape=(None,1),input_var=input)

    
cnn = lasagne.layers.DenseLayer(
            cnn, 
            num_units=4, 
            b=None,
            nonlinearity=None)

cnn = lasagne.layers.ReshapeLayer(cnn,[1,1,2,2])

cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))

test_output=lasagne.layers.get_output(cnn,deterministic=False)
test_loss= T.sum(test_output)

code.interact(local=locals())
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








