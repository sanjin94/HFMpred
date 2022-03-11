import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Dset:
    def __init__(self, ts, train_validation, avg_const):
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
    def tv_split(self, v1, v2, batch):
        tv_index_middle = int((len(v1) * self.train_validation) / batch) * batch
        tv_index_last = int((len(v1) - tv_index_middle) / batch) * batch + tv_index_middle
        v1_train = v1[0:tv_index_middle]
        v2_train = v2[0:tv_index_middle]
        v1_validation = v1[tv_index_middle:tv_index_last]
        v2_validation = v2[tv_index_middle:tv_index_last]
        index_train = [i for i in range(tv_index_middle)]
        index_validation = [i for i in range(tv_index_middle, tv_index_last)]

        return [v1_train, v2_train, index_train, v1_validation, v2_validation, index_validation]


class Dl_model(nn.Module):
    def __init__(self, bsize, nlayers, device, dropout=0.2):
        super(Dl_model, self).__init__()
        self.bsize = bsize
        self.nlayers = nlayers
        self.linear = nn.Linear(
            in_features=self.bsize * nlayers,
            out_features=self.bsize
        )
        self.lstm = nn.LSTM(
            input_size=self.bsize,
            hidden_size=self.bsize,
            num_layers=nlayers
        )
        self.dropout = nn.Dropout(dropout)
        self.h0 = torch.randn(nlayers, 1, self.bsize)
        self.c0 = torch.randn(nlayers, 1, self.bsize)
        self.device = device

    def forward(self, v1):
        torch.cuda.empty_cache()
        tr = v1.view(1, 1, self.bsize)
        device = self.device
        tr = tr.to(device).float()
        h0 = self.h0.to(device).float()
        c0 = self.c0.to(device).float()
        lstm = self.lstm.to(device)
        linear = self.linear.to(device).float()
        x, (h_out, _) = lstm(tr, (h0, c0))
        h_out = self.dropout(h_out.flatten())
        x = linear(h_out)
        return x

class MLP(nn.Module):
    def __init__(self, bsize, nperceptrons, device, dropout):
        super(MLP, self).__init__()
        self.bsize = bsize
        self.nperceptrons = nperceptrons
        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(
            in_features=bsize,
            out_features=bsize
        )

    def forward_mlp(self, v1):
        device = self.device
        linear = self.linear.to(device).float()
        middle = int(len(v1)/2)
        middle_linear = nn.Linear(
            in_features=self.bsize,
            out_features=middle
        ).to(device).float()
        last_linear = nn.Linear(
            in_features=middle,
            out_features=self.bsize
        ).to(device).float()
        x = v1
        x = x.to(device).float()
        x = self.dropout(F.relu(linear(x)))
        #x = F.tanh(middle_linear(x))
        #x = F.tanh(last_linear(x))
        x = linear(x)
        return x

def u_val(delta_t, q_):
    u_list = np.array([])
    for i in range(len(delta_t)):
        n = i + 1
        dt = np.sum(delta_t[0:n]) / n
        q = np.sum(q_[0:n]) / n
        u = -q / dt
        u_list = np.append(u_list, u)

    return u_list, u
