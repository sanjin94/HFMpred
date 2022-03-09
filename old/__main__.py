from os import name
from numpy.core.numeric import full
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math

from torch.nn.modules import dropout

import data

device = torch.device('cuda')

bsize = data.BATCH_SIZE
channels = data.CHANNELS

# generating data
train_validation = data.TV

ti_train = data.ti[0:int(np.round(train_validation*len(data.ti)))]
ti_train = torch.from_numpy(ti_train)
te_train = data.te[0:int(np.round(train_validation*len(data.ti)))]
te_train = torch.from_numpy(te_train)
#train = torch.cat((ti_train, te_train), 1)

# visual check
# #print('\ntrain shape: ', train.shape)

q_train = data.q[0:int(np.round(train_validation*len(data.ti)))]
q_train = torch.from_numpy(q_train)

# visual check
# #print('\nq shape: ', q_train.shape)

ti_validation = data.ti[len(ti_train):len(data.ti)]
ti_validation = torch.from_numpy(ti_validation)
te_validation = data.te[len(ti_train):len(data.ti)]
te_validation = torch.from_numpy(te_validation)
#validation = torch.cat((ti_validation, te_validation), 1)

# visual check
# #print('\nvalidation shape: ', validation.shape)

q_validation = data.q[len(ti_train):len(data.ti)]
q_validation = torch.from_numpy(q_validation)

# visual check
# #print('q valdiation shape : ', q_validation.shape)

# model

class Model(nn.Module):

    # constructor
    def __init__(self, in_=2, L1=64, L2=128, out_=1, ksize=3):
        super(Model, self).__init__()
        self.cnn = nn.Conv1d(in_channels=in_, out_channels=out_,
                             kernel_size=ksize)
        self.lstm1 = nn.LSTM(input_size=(data.BATCH_SIZE-ksize+1),
                             hidden_size=data.BATCH_SIZE, num_layers=16,
                             dropout=0.6)
        self.h0 = torch.randn(16, out_, data.BATCH_SIZE).cuda()
        self.c0 = torch.randn(16, out_, data.BATCH_SIZE).cuda()
        #self.lstm2 = nn.LSTM(input_size=L1, hidden_size=L2, num_layers=2,
        #                     dropout=0.6)
        #self.h1 = torch.randn(2, out_, L2).cuda()
        #self.c1 = torch.randn(2, out_, L2).cuda()
        self.linear = nn.Linear(in_features=data.BATCH_SIZE, out_features=data.BATCH_SIZE)

    # forward pass
    def forward(self, train):
        x = self.cnn(train)
        x = torch.relu(x)
        x, (hn0, cn0) = self.lstm1(x, (self.h0, self.c0))
        #print(x.shape)
        #x, (hn1, cn1) = self.lstm2(x, (self.h1, self.c1))
        #print(x.shape)
        x = self.linear(x)
        x = torch.squeeze(x)
        #print(x.shape)
        return x


    # data manipulation
    def data_rearange(self, ti_train, te_train, ti_validation, channels,
                      te_validation, q_train, q_validation, bsize):
        nbatches_t = ti_train.shape[0]/bsize
        nbatches_v = ti_validation.shape[0]/bsize

        training = torch.randn(math.ceil(nbatches_t), channels, bsize)
        validation = torch.randn(math.ceil(nbatches_v), channels, bsize)
        q_t = torch.randn(math.ceil(nbatches_t), bsize)
        q_v = torch.randn(math.ceil(nbatches_v), bsize)

        if int(nbatches_t) < nbatches_t:
            nbatches_t = int(nbatches_t + 1)
            full_tensor_train = False
        else:
            nbatches_t = int(nbatches_t)
            full_tensor_train = True

        if int(nbatches_v) < nbatches_v:
            nbatches_v = int(nbatches_v + 1)
            full_tensor_validation = False
        else:
            nbatches_t = int(nbatches_t)
            full_tensor_train = True


        if full_tensor_train:
            for i in range(nbatches_t):
                for j in range(bsize):
                    training[i, 0, j] = ti_train[i*bsize+j][0]
                    training[i, 1, j] = te_train[i*bsize+j][0]
                    q_t[i, j] = q_train[i*bsize+j][0]
        else:
            for i in range(nbatches_t-1):
                for j in range(bsize):
                    training[i, 0, j] = ti_train[i*bsize+j][0]
                    training[i, 1, j] = te_train[i*bsize+j][0]
                    q_t[i, j] = q_train[i*bsize+j][0]
            for i in range(((nbatches_t-1)*bsize-ti_train.shape[0])):
                training[nbatches_t-1, 0, i] = ti_train[(nbatches_t-1)*bsize + i][0]
                training[nbatches_t-1, 1, i] = te_train[(nbatches_t-1)*bsize + i][0]
                q_t[nbatches_t-1, i] = q_train[(nbatches_t-1)*bsize + i][0]

        if full_tensor_validation:
            for i in range(nbatches_v):
                for j in range(bsize):
                    validation[i, 0, j] = ti_validation[i*bsize+j][0]
                    validation[i, 1, j] = te_validation[i*bsize+j][0]
                    q_v[i, j] = q_validation[i*bsize+j][0]
        else:
            for i in range(nbatches_v-1):
                for j in range(bsize):
                    validation[i, 0, j] = ti_validation[i*bsize+j][0]
                    validation[i, 1, j] = te_validation[i*bsize+j][0]
                    q_v[i, j] = q_validation[i*bsize+j][0]
            for i in range((nbatches_v-1)*bsize - ti_validation.shape[0]):
                validation[nbatches_v-1, 0, i] = ti_validation[(nbatches_v-1)*bsize + i][0]
                validation[nbatches_v-1, 0, i] = ti_validation[(nbatches_v-1)*bsize + i][0]
                q_v[nbatches_v-1, 0, i] = q_validation[(nbatches_v-1)*bsize + i][0]
        return training, validation, q_t, q_v

    # training
def train_model(model, train_loader, validation_loader, q_train_loader,
                q_validation_loader, optimizer, criterion, n_epochs):
    loss_list = []
    validation_loss_list = []
    model.to(device)
    train_loader = train_loader.cuda()
    validation_loader = validation_loader.cuda()
    q_train_loader = q_train_loader.cuda()
    q_validation_loader = q_validation_loader.cuda()

    for epoch in range(n_epochs):

        # Training
        #for x in train_loader:
        #:w
        # print('\n epoch ', epoch, 'out of ', n_epochs)
        model.train()
        optimizer.zero_grad()
        z = model.forward(train_loader)
        #print(z.shape)
        #print(q_train_loader.shape)
        loss = criterion(z, q_train_loader)
        print('\n epoch ', epoch, 'out of ', n_epochs, 'loss: ', loss.cpu().detach().numpy())
        loss.backward()
        optimizer.step()
        loss_list.append(loss.data)

        # Validation
        with torch.set_grad_enabled(False):
            #for x_test in validation_loader:
            model.eval()
            z_val = model.forward(validation_loader)
            loss = criterion(z_val, q_validation_loader)
            validation_loss_list.append(loss.data)

    print(z.shape)
    print(z_val.shape)
    z = torch.cat((z, z_val), 0)
    z = torch.squeeze(z.cpu().detach(), 0)

    return loss_list, validation_loss_list, z

# execution of the script
if __name__ == '__main__':
    dl_model = Model()
    tr, val, q_tr, q_val = dl_model.data_rearange(ti_train, te_train, ti_validation, channels,
                                                  te_validation, q_train, q_validation, bsize)
    # visual check
    #print(tr.shape, val.shape, q_tr.shape, q_val.shape)

    training_loss, validation_loss, z = train_model(dl_model, tr, val, q_tr, q_val,
                                                 torch.optim.Adam(dl_model.parameters(), lr = 0.1),
                                                 nn.MSELoss(), 1000)
    #plt.plot(validation_loss)
    #plt.plot(training_loss)
    plt.plot(z)
    plt.show()
