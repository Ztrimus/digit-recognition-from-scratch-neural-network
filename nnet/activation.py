import torch

def sigmoid(z):
    """
    function:: sigmoid(Tensor)
    Calculates sigmoid values for tensors
            sigmoid(x) = 1/(1+e^-x)
    Args:
        z (Tensor) : the input tensor
    Example::
        >>> a = torch.arange(5, dtype = torch.float)
        >>> a
        tensor([0., 1., 2., 3., 4.])
        >>> sigmoid(a)
        tensor([0.5000, 0.7311, 0.8808, 0.9526, 0.9820])
    Return::
        <class 'torch.Tensor'>
    """
    z_tensor = torch.Tensor.float(z)
    result =   1 / (1 + torch.exp(-z_tensor))
    return result

def delta_sigmoid(z):
    """
    function:: sigmoid(array/Tensor)
    Calculates derivative of sigmoid function
            delta_sigmoid(x) = sigmoid(x)*(1 - sigmoid(x))
    Args:
        z (Tensor) : the input tensor
    Required:
        function:: sigmoid(z)
    Example::
        >>> a = torch.arange(5, dtype = torch.float)
        >>> a
        tensor([0., 1., 2., 3., 4.])
        >>> delta_sigmoid(a)
        tensor([0.2500, 0.1966, 0.1050, 0.0452, 0.0177])
    Return::
        <class 'torch.Tensor'>
    """
    z_tensor = torch.Tensor.float(z)
    grad_sigmoid =  sigmoid(z_tensor)*(1 - sigmoid(z_tensor))
    return grad_sigmoid

def softmax(x):
    """
    function:: softmax(array/Tensor)
    Calculates stable softmax (minor difference from normal softmax) values for tensors
            softmax(x) = C*e^x / sum(C*e^x)
                       = e^(x + log(C)) / sum(e^(x + log(C)))
            log(C) = - max(x)
    Args:
        x (Tensor) : the input tensor
    Example::
        >>> a = torch.arange(5, dtype = torch.float)
        >>> a
        tensor([0., 1., 2., 3., 4.])
        >>> softmax(a)
        tensor([0.0117, 0.0317, 0.0861, 0.2341, 0.6364])
    Return::
        <class 'torch.Tensor'>

    """
    stable_softmax = torch.zeros(x.size())
    for idx, i in enumerate(x):
        stable_softmax[idx] = torch.exp(i) / sum(torch.exp(i))
    return stable_softmax

if __name__ == "__main__":
    pass
