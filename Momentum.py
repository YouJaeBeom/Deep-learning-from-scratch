import numpy as np

class Momentum:
    def __init__(self,lr=0.001,momentum=0.9):
        self.lr =lr
        self.monentum=momentum
        self.v=None

    def update(self,params,grads):
        if self.v is None:
            self.v={}
            for key,val in params.items():
                self.v[key]=np.zeros_like(val)

        for key in params.keys():
            self.v[key]=self.monentum*self.v[key]-self.lr*grads[key]
            params[key]=params[key]+self.v[key]