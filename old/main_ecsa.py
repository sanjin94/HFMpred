from os import name
from numpy.core.numeric import full
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from torch.nn.modules import dropout

import data

torch.manual_seed(1)
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

class Nnet(nn.Module):
    def __init__(self, in_=2, hidden_=3, out_=1):
        super(Nnet, self).__init__()
        self.hidden = nn.Linear(in_, hidden_)
        self.predict = nn.Linear(hidden_, out_)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
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


def train_model(model, train_loader, validation_loader, q_train_loader,
                q_validation_loader, optimizer, criterion, n_epochs):
    validation_loss_list = []
    loss_list = []
    device = torch.device('cuda')
    model.to(device)
    train_loader = train_loader.cuda()
    validation_loader = validation_loader.cuda()
    q_train_loader = q_train_loader.cuda()
    q_validation_loader = q_validation_loader.cuda()

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        z = model.forward(train_loader)
        loss = criterion(z, q_train_loader)
        print('\n epoch ', epoch, 'out of ', n_epochs, 'loss: ', loss.cpu().detach().numpy())
        loss.backward()
        optimizer.step()
        loss_list.append(loss.data)

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
        

if __name__ == '__main__':
    model_ = Nnet(1,3,1)
    optimizer = torch.optim.Adam(model_.parameters())
    loss_func = torch.nn.MSELoss()
    n_epochs = 2000

    tr, val, q_tr, q_val = model_.data_rearange(ti_train, te_train, ti_validation, channels, 
                                                  te_validation, q_train, q_validation, bsize)

    training_loss, validation_loss, z = train_model(model_, tr, val, q_tr, q_val, 
                                                   optimizer, loss_func, n_epochs)
    
    plt.plot(z)
    plt.show()