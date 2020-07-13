import numpy as np
class AffineLayer:
    def __init__(self,W,b):
        self.x=None
        self.W=W
        self.b=b
        self.dW=None
        self.db=None

    def forward(self,x):
        self.x=x
        out=np.dot(x,self.W)+self.b

        return out

    def backward(self,dout):
        dx=np.dot(dout,self.W.T)
        self.dW=np.dot(self.x.T,dout)
        self.db=np.sum(dout,axis=0)

        return dx
