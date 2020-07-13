class Relu:
    def __init__(self):
        # mask는 bool로 구성된 numpy
        self.mask=None

    def forward(self,x):
        # 0이하 true
        # 0이상 false
        self.mask=(x<=0)
        out=x.copy()
        out[self.mask]=0

        return out

    def backward(self,dout):
        # 0이하 역전파 0
        dout[self.mask]=0
        dx=dout

        return dx

