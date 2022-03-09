from math import ceil
import torch
from torch import nn
from torch.nn.modules import dropout
import numpy as np


class Model(nn.Module):
    # delta_t -- input vector, q -- output vector, bdiv -- batches division
    def __init__(self, delta_t_init, q_init, bsize, nlayers, device, dropout):
        super(Model, self).__init__()
        self.bsize = bsize
        self.bdiv = int(len(delta_t_init) / bsize)
        # neglecting last len(q) - bsize * bdiv elements
        self.delta_t = delta_t_init[0:self.bsize * self.bdiv].clone().detach()
        self.q = q_init[0:self.bsize * self.bdiv].clone().detach()
        self.bdiv = self.bdiv
        self.lstm = nn.LSTM(input_size=self.bsize,
                            hidden_size=self.bsize, num_layers=nlayers,
                            dropout=dropout)
        self.h0 = torch.randn(nlayers, 1, self.bsize)
        self.c0 = torch.randn(nlayers, 1, self.bsize)
        self.linear = nn.Linear(in_features=self.bsize*nlayers,
                                out_features=self.bsize)
        self.device = device
        self.nlayers = nlayers

    def forward(self, tr):
        tr = tr.view(1, 1, self.bsize)
        device = self.device
        tr = tr.to(device).float()
        h0 = self.h0.to(device).float()
        c0 = self.c0.to(device).float()
        lstm = self.lstm.to(device)
        linear = self.linear.to(device).float()
        x, (h_out, _) = lstm(tr, (h0, c0))
        h_out = h_out.flatten()
        x = linear(h_out)
        return x


class Dset:
    def __init__(self, ts, train_validation, avg_const=1):
        self.head = ts.columns.values
        self.ts = ts
        self.avg_const = avg_const
        self.train_validation = train_validation

    # averaging the timeseries
    def ts_average(self):
        c = self.avg_const
        v1 = self.ts[self.head[0]].copy()  # in my case delta_t
        v2 = self.ts[self.head[1]].copy()  # in my case q

        itr = int(len(v1)/c)
        avg_list = []

        v1_avg = np.array([])
        v2_avg = np.array([])

        for j in range(itr):
            ##new_v1 = probaj hoće ubrzat da definiraš vrijednos pa onda appendaš
            v1_avg = np.append(v1_avg, np.sum(v1[j*c:(j+1)*c])/c)
            v2_avg = np.append(v2_avg, np.sum(v2[j*c:(j+1)*c])/c)

        return v1_avg, v2_avg

    # train - validation split
    def tv_split(self, v1, v2):
        limit = int(len(v1) * self.train_validation)
        end = len(v1)
        v1_train = v1[0:limit]
        v2_train = v2[0:limit]
        v1_validation = v1[limit:end]
        v2_validation = v2[limit:end]
        first = []
        last = []
        for i in range(limit):
            first.append(i)
        for i in range(limit, len(v1)):
            last.append(i)

        return [v1_train, v2_train, first, v1_validation, v2_validation, last]
