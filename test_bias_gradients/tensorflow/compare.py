import numpy as np
a=np.load("theano_conv1.npy") 
b=np.load("tensor_first_conv_res.npy")
c=np.transpose(b,(0,3,1,2))

