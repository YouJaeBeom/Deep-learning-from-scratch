import numpy as np
class AdaGrad:
    def __init__(self,lr=0.01):
        self.lr=lr
        self.h=None

    def update(self,params,grads):
        if self.h is None:
            self.h={}
            for key,val in params.items():
                self.h[key]=np.zero_like(val)

        for key in params.keys():
            self.h[key]+=grads[key]*grads[key]
            # 1e-7 -> self.h에 0이 담겨있다해도 0으로 나누어주는 사태를 막아준다.
            params[key]-=self.lr*grads[key]/(np.sqrt(self.h[key])+1e-7)