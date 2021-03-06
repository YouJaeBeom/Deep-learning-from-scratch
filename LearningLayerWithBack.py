import sys, os
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import orderdict
from AffineLayer import Affine
from Relu import Relu
from SoftmanxWithLoss import SoftmaxWithLoss

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def cross_entropy_error(y,t):
    if y.nidm==1:
        t=t.reshape(1,t.size)
        y=y.reshape(1,y.size)
    batch_size=y.shape[0]
    return -np.sum(t*np.log(y+1e-7))/batch_size
def softmax(x):
    c=np.max(x)
    exp_a=np.exp(x-c)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    return y
def sigmoid(x):
    return 1/(1+np.exp(-x))
## gradient function
def numerical_gradient(f,x):
    h=1e-4
    grad=np.zeros_like(x)

    for idx in range(x.size):
        tmp_val=x[idx]

        #f(x+h) 계산
        x[idx]=tmp_val+h
        fxh1=f(x)

        # f(x-h) 계산
        x[idx]=tmp_val-h
        fxh2=f(x)

        grad[idx]=(fxh1-fxh2)/(2*h)

        return grad

class LayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        #계층 생성
        self.layers=orderdict()
        self.layers['Affine1']=Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1']=Relu()
        self.layers['Affine2']=Affine(self.params['W2'], self.params['b2'])

        self.lastLayer=SoftmaxWithLoss()



    def predict(self, x):
        for layer in self.layers.values():
            x=layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)

        return self.lastLayer.forward(y, t)

    def accuracy(self,x,t):
        y=self.predict(x)
        y=np.argmax(y,axis=1)
        if t.ndim != 1: t=np.argmax(t,axis=1)
        accuracy = np.sum(y==t)/float(x.shape[0])
        return accuracy

    def numerical_gradient(self,x,t):
        loss_W=lambda W: self.loss(x,t)

        grads={}
        grads['W1'] = numerical_gradient(loss_W,self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self,x,t):
        self.loss(x,t)

        #역전파
        dout=1
        dout=self.lastLayer.backward(dout)

        layers= list(self.layers.values)
        layers.reverse()
        for layer in layers:
            dout=layer.backward(dout)

        grads={}
        grads['W1']=self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads



(x_train,t_train),(x_test,t_test)=mnist(normalize=True,one_hot_label=True)

train_loss_list=[]
train_acc_list=[]
test_acc_list=[]

iters_num=10000
train_size = x_train.shape[0]
batch_size=100
learning_rate=0.1
iter_per_epoch=max(train_size/batch_size,1)
network=LayerNet(input_size=784,hidden_size=50,output_size=10)

for i in range(iters_num):
    batch_mask=np.random.choice(train_size,batch_size)
    x_batch=x_train[batch_mask]
    t_batch=t_train[batch_mask]

    grad=network.gradient(x_batch,t_batch)

    for key in ('W1','b1','W2','b2'):
        network.params[key]-=learning_rate*grad[key]

    loss=network.loss(x_batch,t_batch)
    train_loss_list.append(loss)
    if i%iter_per_epoch==0:
        train_acc=network.accuracy(x_train,t_train)
        test_acc=network.accuracy(x_test,t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test_acc",str(train_acc)," ",str(test_acc))







