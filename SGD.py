from LearningLayerWithBack import LayerNet


class SGD:
    def __init__(self,lr=0.01):
        self.lr=lr

    def update(self,params,grads):
        for key in params.keys():
            ## SGD  확률적 경사 하강법
            params[key]-=self.lr*grads[key]



network=LayerNet()
optimizer=SGD()

for i in range(10000):
    ## ....
    grads=network.numerical_gradient(x_batch,t_batch)
    params=network.params
    ## 값을 넘겨줄떄 각 파라미터와 기울기값을 넘겨주면 optimization실행 ㅡㅐㅡ두셔
    optimizer.update(params,grads)