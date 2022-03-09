import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch

import model_classes as dlm

torch.manual_seed(1618)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dropout = 0.7

#importing data
q_z21 = pd.read_csv('model_dev/test/q_zagreb_21.csv', sep=',')
dt_z = pd.read_csv('model_dev/test/Zagreb2021.csv', sep=',')
data = pd.concat([dt_z, q_z21], axis=1) #pd.read_csv('model_dev/dataset_1min_lenti.csv', sep=',')

data_worker = dlm.Dset(data, train_validation=1, avg_const=1)
delta_t_avg, q_avg = data_worker.ts_average()
train_validation = data_worker.tv_split(delta_t_avg, q_avg)
delta_t_train = train_validation[0][48:len(train_validation[0])]
q_train = train_validation[1][48:len(train_validation[0])]

#delta_t = np.array(data['delta_t'].values)
delta_t = torch.tensor(delta_t_train).float()
#q = np.array(data['q'].values)
q = torch.tensor(q_train).float()
bsize = 48

dl_model = dlm.Model(delta_t, q, bsize=bsize, nlayers=32, device=device, dropout=dropout)

#dt_b = dl_model.prep_fwd()

#fwd_output = dl_model.forward(dt_b)

#### training
num_epochs = 200
learning_rate = 1e-4
q = q.detach()[0:dl_model.bsize*dl_model.bdiv].view(dl_model.bdiv, 1, dl_model.bsize).to(device)
delta_t = delta_t.detach()[0:dl_model.bsize*dl_model.bdiv].view(dl_model.bdiv, 1, dl_model.bsize)

criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(dl_model.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

# Train the model
q_test = q.detach().flatten().cpu().numpy()

for epoch in range(num_epochs):
  out = []

  for batch in range(dl_model.bdiv):
    optimizer.zero_grad()
    outputs = dl_model.forward(delta_t[batch][0])
    out = np.append(out, outputs.detach().cpu().numpy())

    # obtain the loss function
    loss = criterion(outputs, q[batch][0])

    loss.backward()

    optimizer.step()

  loss_dset = (np.square(out - q_test)).mean()
  if epoch % 1 == 0:
    print("Epoch: %d, loss: %1.5f" % (epoch + 1, loss_dset.item()))

## prediction part
delta_t_val = torch.tensor(train_validation[3][0:dl_model.bsize*dl_model.bdiv])
q_val = torch.tensor(train_validation[4][0:dl_model.bsize*dl_model.bdiv])

dl_model.eval()

with torch.no_grad():
  prediction = []
  bdiv = int(len(delta_t.flatten())/bsize)
  for batch in range(bdiv):
    prediction_b = dl_model(delta_t[batch][0]).cpu().detach().numpy()
    prediction = np.append(prediction, prediction_b)

plt.plot(prediction)
plt.plot(q_test)
plt.show()

torch.save(dl_model, 'model_dev/test/model/32lstm.pt')

data_val = pd.read_csv('model_dev/test/dataset_1min_izolacija.csv', sep=',')
data_worker_val = dlm.Dset(data_val, train_validation=1, avg_const=30)
delta_t_avg, q_avg = data_worker_val.ts_average()
train_validation = data_worker_val.tv_split(delta_t_avg, q_avg)
bdiv = int(len(train_validation[0]) / bsize)
delta_t_validation = train_validation[0][0:bdiv]
q_validation = train_validation[1][0:bdiv]

plt.plot(delta_t_validation)
plt.plot(q_validation)
plt.show()

#delta_t = np.array(data['delta_t'].values)
delta_t_val = torch.tensor(delta_t_validation).float().view(bdiv, 1, bsize)
#q = np.array(data['q'].values)
q_val = torch.tensor(q_validation).float().view(bdiv, 1, bsize)
#q_test = pd.read_csv('model_dev/test/komora_izolacija_q.csv', sep=',', header='q')

with torch.no_grad():
  prediction = []
  for batch in range(bdiv):
    prediction_validation = dl_model(delta_t_val).cpu().detach().numpy()
    prediction = np.append(prediction, prediction_validation)

plt.plot(prediction)
plt.plot(q_validation)

plt.show()
