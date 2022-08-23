import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# introduce gpu computing
device = torch.device('cuda')

m = nn.Conv1d(16, 33, 3)
inp = torch.randn(20, 16, 50)
output = m(inp)

print('\n Output tensor: \n', output)
print('\n Shape of input tensor: ', inp.shape)
print('\n Shape of output tensor: ', output.shape)