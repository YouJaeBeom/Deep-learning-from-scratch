import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

######################
## 가중치의 초기값을 그대로
######################
x=np.random.rand(1000,100)
node_num=100 # 은닉층 노드 수
hidden_layer_size=5 #은닉층 수
activations={}

for i in range(hidden_layer_size):
    if i!=0:
        x=activations[i-1]

    w=np.random.randn(node_num,node_num)*1
    a=np.dot(x,w)
    z=sigmoid(a)
    activations[i]=z

## histogram
for i,a in activations.items():
    plt.subplot(1,len(activations),i+1)
    plt.title(str(i+1)+"-layer")
    plt.hist(a.flatten(),30,range=(0,1))
plt.show()
## -> 0과 1에 값이 치우처있다. 기울기 값이 점점 작아지다가 사라지게된다.


######################
## 가중치의 초기값을 0.01
######################
x=np.random.rand(1000,100)
node_num=100 # 은닉층 노드 수
hidden_layer_size=5 #은닉층 수
activations={}

for i in range(hidden_layer_size):
    if i!=0:
        x=activations[i-1]

    w=np.random.randn(node_num,node_num)*0.01
    a=np.dot(x,w)
    z=sigmoid(a)
    activations[i]=z

## histogram
for i,a in activations.items():
    plt.subplot(1,len(activations),i+1)
    plt.title(str(i+1)+"-layer")
    plt.hist(a.flatten(),30,range=(0,1))
plt.show()
## -> 활성화값들이 치우쳐있다 -> 표현력관점에서 큰문제

## 각 층의 활성화값들은 골고루 분포되어야한다.