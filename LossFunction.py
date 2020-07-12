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

