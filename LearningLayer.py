import numpy as np

## Loss function
# 1. MSE
def mean_squared_error(y,t):
    return 0.5*np.sum((y-t)**2)

# 2.cross entropy
def cross_entropy_error(y,t):
    delta=1e-7
    return -np.sum(t*np.log(y+delta))

## minibatch cross entropy -  one-hot encoding
def cross_entropy_error(y,t):
    if y.nidm==1:
        t=t.reshape(1,t.size)
        y=y.reshape(1,y.size)
    batch_size=y.shape[0]
    return -np.sum(t*np.log(y+1e-7))/batch_size

## minibatch cross entropy - not one-hot encoding
def cross_entropy_error(y,t):
    if y.nidm==1:
        t=t.reshape(1,t.size)
        y=y.reshape(1,y.size)
    batch_size=y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size),t]+1e-7))/batch_size

## numerical differentation 수치 미분
def numerical_diff(f,x):
    h=1e-4
    return (f(x+h)-f(x-h))/(2*h)

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

## 경사하강법
def gradient_descent(f,init_x,lr,step_num=100):
    x=init_x

    for i in range(step_num):
        grad=numerical_gradient(f,x)
        x -= lr* grad
    return x

def function_2(x):
    return x[0]**2+x[1]**2
init_x=np.array([-3.0,4.0])
print(gradient_descent(function_2,init_x=init_x,lr=0.1,step_num=100))
print(gradient_descent(function_2,init_x=init_x,lr=10.0,step_num=100))
print(gradient_descent(function_2,init_x=init_x,lr=1e-10,step_num=100))
