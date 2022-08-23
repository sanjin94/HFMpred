import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# introduce gpu computing
device = torch.device('cuda')

# defining input data (20 batches, 16 channels, 50 elements in channel)
input_ = torch.randn(20, 16, 50)
true_output_train = torch.randn(20, 16, 1)
input_ = input_.cuda()
true_output_train = true_output_train.cuda()

# model class
class Convolution(nn.Module):
    # constructor
    def __init__(self, in_=16, out_=16, kernel_=3):
        super(Convolution, self).__init__()
        self.cnn = nn.Conv1d(in_, out_, kernel_size=kernel_)
        self.linear = nn.Linear(in_features=48, out_features=1)

    # convolution application
    def forward(self, input_):
        x = self.cnn(input_)
        x = torch.relu(x)
        x = self.linear(x)
        x = torch.relu(x)
        
        return x

convolution_ = Convolution()

# training 
def train_model(model, input_train, true_output_train,
                optimizer, criterion, n_epochs):
    loss_list = []
    model.to(device)

    for epoch in range(n_epochs):
        # training
        model.train()
        optimizer.zero_grad()
        z = model.forward(input_train)
        loss = criterion(z, true_output_train)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.data)

    return loss_list

training_loss = train_model(convolution_, input_, true_output_train,
                            torch.optim.Adam(convolution_.parameters(), lr = 0.1),
                            nn.MSELoss(), 1000)

fig = plt.figure()
plt.plot(training_loss)
fig.savefig('fig.png', dpi=300)