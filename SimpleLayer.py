import numpy as np

# 활성화함수
# 1. sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# 2. Relu function
def relu(x):
    return np.maximum(0,x)

## 출력층의 활성화 함수
# 1. 항등함수  -> 회귀
def identity_function(x):
    return x

# 2. softmax function -> 분류
def init_network():
    network={}
    network['W1']=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1']=np.array([0.1,0.2,0.3])
    network['W2']=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2']=np.array([0.1,0.2])
    network['W3']=np.array([[0.1,0.3],[0.2,0.4]])
    network['b3']=np.array([0.1,0.2])

    return network

## 활성화 함수 sigmoid
def forward1(network,x):
    W1,W2,W3=network['W1'],network['W2'],network['W3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']

    a1 = np.dot(x,W1)+b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y=identity_function(a3)

    return y

## 활성화 함수 relu
def forward2(network,x):
    W1,W2,W3=network['W1'],network['W2'],network['W3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']

    a1 = np.dot(x,W1)+b1
    z1 = relu(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = relu(a2)
    a3 = np.dot(z2, W3) + b3
    y=identity_function(a3)

    return y

network=init_network()
x=np.array([1.0,0.5])
y1=forward1(network,x)
print(y1)
y2=forward2(network,x)
print(y2)
