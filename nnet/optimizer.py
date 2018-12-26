import torch

def mbgd(weights, biases, dw1, db1, dw2, db2, dw3, db3, lr):
    """Mini-batch gradient descent
    """
    new_w1 = weights['w1'] - lr * dw1
    new_w2 = weights['w2'] - lr * dw2
    new_w3 = weights['w3'] - lr * dw3
    new_b1 = biases['b1'] - lr * db1
    new_b2 = biases['b2'] - lr * db2
    new_b3 = biases['b3'] - lr * db3
    new_weights = {'w1': new_w1, 'w2': new_w2, 'w3': new_w3}
    new_biases = {'b1': new_b1, 'b2': new_b2, 'b3': new_b3}

    return new_weights, new_biases

if __name__ == "__main__":
    pass
