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
# overflow 발생
def softmax(x):
    exp_a=np.exp(x)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    return y
# overflow 문제로 수정
def softmax_modify(x):
    c=np.max(x)
    exp_a=np.exp(x-c)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    return y



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

## 활성화 함수 sigmoid + softmax
def forward3(network,x):
    W1,W2,W3=network['W1'],network['W2'],network['W3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']

    a1 = np.dot(x,W1)+b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y=softmax_modify(a3)

    return y

## 활성화 함수 relu + softmax
def forward4(network,x):
    W1,W2,W3=network['W1'],network['W2'],network['W3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']

    a1 = np.dot(x,W1)+b1
    z1 = relu(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = relu(a2)
    a3 = np.dot(z2, W3) + b3
    y=softmax_modify(a3)

    return y

network=init_network()
x=np.array([1.0,0.5])
y1=forward1(network,x)
print("sigmoid + identity",y1)
y2=forward2(network,x)
print("relu + identity",y2)
y3=forward3(network,x)
print("sigmoid + softmax",y3)
y4=forward4(network,x)
print("relu + softmax",y4)
