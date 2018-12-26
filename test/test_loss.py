from context import nnet
from nnet import loss, activation

import unittest
import torch
import math
import numpy as np

class TestLossModule(unittest.TestCase):

    def test_cross_entropy(self):
        # settings
        batch_size = 11
        N_out = 12

        # tensors
        outputs = torch.rand((batch_size, N_out), dtype=torch.float)
        labels = torch.randint(high=N_out, size=(batch_size,), dtype=torch.long)

        creloss = loss.cross_entropy_loss(activation.softmax(outputs), labels)
        assert type(creloss) == float
        # write more robust and rigourous test cases here
        nll = torch.nn.functional.cross_entropy(outputs, labels)
        self.assertAlmostEqual(creloss, nll.item(), places=6)
    
    def test_delta_cross_entropy_loss(self):
        # settings
        batch_size = 11
        N_out = 17
        precision = 0.000001

        # tensors
        outputs = torch.rand((batch_size, N_out), dtype=torch.float, requires_grad=True)
        labels = torch.randint(high=N_out, size=(batch_size,), dtype=torch.long)

        # calculate gradients from scratch
        grads_creloss = loss.delta_cross_entropy_softmax(activation.softmax(outputs), labels)
        
        # calculate gradients with autograd
        nll = torch.nn.functional.cross_entropy(outputs, labels)
        nll.backward()

        assert isinstance(grads_creloss, torch.FloatTensor)
        assert grads_creloss.size() == torch.Size([batch_size, N_out])
        self.assertTrue(torch.le(torch.abs(grads_creloss - outputs.grad), precision).all())
        # write more robust test cases here
        # you should write gradient checking code here

if __name__ == '__main__':
    unittest.main()
