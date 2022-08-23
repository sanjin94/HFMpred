import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

#device = torch.device('cuda')

#input_ = torch.randn(19, 2, 28)
#input_ = input_.cuda()

#class LSTM(nn.Module):
    #def __init__(self, in_= , out_= , kernel_=)


################ Primjer s neta ###########################
rnn = nn.LSTM(10, 16, 2)
input_ = torch.randn(5, 3, 10)
print(input_)
print(input_.shape)
h0 = torch.randn(2, 3, 16)
print(h0)
c0 = torch.randn(2, 3, 16)
print(c0)
output, (hn, cn) = rnn(input_, (h0, c0))
print(output.shape)
rnn2 = nn.LSTM(16, 32, 3)
h1 = torch.randn(3, 3, 32)
c1 = torch.randn(3, 3, 32)
output2, (hn1, cn1) = rnn2(output, (h1, c1))
print(output2)
print(output2.shape)