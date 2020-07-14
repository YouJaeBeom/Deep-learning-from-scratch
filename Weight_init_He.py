import numpy as np
import matplotlib.pyplot as plt

# 2. Relu function
def relu(x):
    return np.maximum(0,x)

######################
## 가중치의 초기값을 He
######################
x=np.random.rand(1000,100)
node_num=100 # 은닉층 노드 수
hidden_layer_size=5 #은닉층 수
activations={}

for i in range(hidden_layer_size):
    if i!=0:
        x=activations[i-1]
    #w = np.random.randn(node_num, node_num) * 0.01
    #w = np.random.randn(node_num, node_num) / np.sqrt(node_num) #xavier사용 -> 층이 깊어질수록 값이 치우쳐진다.
    w=np.random.randn(node_num,node_num)*(np.sqrt(2)/np.sqrt(node_num)) #HE초기값 골고루 분포
    a=np.dot(x,w)
    z=relu(a)
    activations[i]=z

## histogram
for i,a in activations.items():
    plt.subplot(1,len(activations),i+1)
    plt.title(str(i+1)+"-layer")
    plt.hist(a.flatten(),30,range=(0,1))
plt.show()
## -> 활성화값들이 치우쳐있다 -> 표현력관점에서 큰문제

## 각 층의 활성화값들은 골고루 분포되어야한다.