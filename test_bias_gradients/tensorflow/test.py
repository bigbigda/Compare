import numpy as np
import theano
import theano.tensor as T

from theano.scalar.basic import UnaryScalarOp, same_out_nocomplex
from theano.tensor.elemwise import Elemwise

class Round3(UnaryScalarOp):
    def c_code(self, node, name, (x,), (z,), sub):
        return "%(z)s = round(%(x)s);" % locals()
    
    def grad(self, inputs, gout):
        (gz,) = gout
        return gz, 

   

round3_scalar = Round3(same_out_nocomplex, name='round3')
round3 = Elemwise(round3_scalar)

def hard_sigmoid(x):
    return T.clip((x+1.)/2.,0,1)

def binary_tanh_unit(x):
    return 2.*round3(hard_sigmoid(x))-1.

input = T.dscalar('inputs')

y=binary_tanh_unit(input)

gy=T.grad(y,input)
import code
code.interact(local=locals())  
fn=theano.function([input],[y,gy])

#myi=np.float32()
#res,rgy=fn(myi)
  

