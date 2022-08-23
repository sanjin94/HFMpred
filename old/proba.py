import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import data

device = torch.device('cuda')

# Generating data
train_validation = data.TV

te_train = data.te[0:int(np.round(train_validation*len(data.ti)))]
te_train = torch.from_numpy(te_train)
q_train = data.q[0:int(np.round(train_validation*len(data.ti)))]
q_train = torch.from_numpy(q_train)

te_validation = data.te[len(te_train):len(data.ti)]
te_validation = torch.from_numpy(te_validation)
q_validation = data.q[len(te_train):len(data.ti)]
q_validation = torch.from_numpy(q_validation)

# Model
class Model(nn.Module):

    # Constructor
    def __init__(self, in_=1, H1=64, H2=64, H3=128, out_=1, drop=0.2):
        super(Model, self).__init__()
        self.drop_ = nn.Dropout(drop)
        self.cnn = nn.Conv1d(in_channels=in_, out_channels=H1, kernel_size=3)
        self.lstm1 = nn.LSTM(input_size=H1, hidden_size=H2, num_layers=1, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=H2, hidden_size=H3, num_layers=1, bidirectional=True)
        self.linear = nn.Linear(in_features=H3, out_features=out_)

    # Prediction
    def forward(self, train):
        x = self.cnn(train)
        x = torch.relu(x)
        x = self.drop_(x)
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.linear(x)
        return x

model1 = Model() # Defining the object of class Model()

## Training of the model model1
train_loader = torch.utils.data.DataLoader(dataset=te_train, batch_size=8)
validation_loader = torch.utils.data.DataLoader(dataset=te_validation, batch_size=8)

# Training function
def train_model(model, train_loader, validation_loader, true_value_train,
                true_value_validation, optimizer, criterion, n_epochs):
    loss_list = []
    validation_loss_list = []
    model.to(device)


    for epoch in range(n_epochs):

        # Training
        for x in train_loader:
            x.to(device)
            model.train()
            optimizer.zero_grad()
            z = model(x)
            true_value_train.to(device)
            loss = criterion(z, true_value_train)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.data)

        # Validation
        with torch.set_grad_enabled(False):
            for x_test in validation_loader:
                model.eval()
                z = model(x_test)
                loss = criterion(z, true_value_validation)
                validation_loss_list.append(loss.data)

    return loss_list, validation_loss

training_loss, validation_loss = train_model(model1, train_loader,
                                             validation_loader, q_train, q_validation,
                                             torch.optim.Adam(model1.parameters(), lr = 0.1),
                                             nn.MSELoss(), 1000)