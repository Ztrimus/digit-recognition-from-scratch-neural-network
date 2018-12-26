import torch
from nnet.activation import *

def cross_entropy_loss(outputs, labels):
    """Calculates cross entropy loss given outputs and actual labels
    H(y,p)=−∑iyilog(pi)
    """    
    m = labels.shape[0]
    log_likelihood = -torch.log(outputs[range(m),labels])
    creloss = torch.sum(log_likelihood) / m    

    return creloss.item()   # should return float not tensor

def delta_cross_entropy_softmax(outputs, labels):
    """
    Calculates derivative of cross entropy loss (C) w.r.t. weighted sum of inputs (Z). 
    """
    m = labels.shape[0]
    outputs[range(m),labels] -= 1
    avg_grads = outputs/m
    return avg_grads

if __name__ == "__main__":
    pass
