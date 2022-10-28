import numpy as np


def sigmoid(x):
    return np.exp(x)/(np.exp(x) + 1)




def deriv_sigmoid(x):
    return np.exp(x)/(1 + np.exp(x))**2
