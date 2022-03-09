import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch

import model_classes as dlm

torch.manual_seed(1618)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dropout = 0.7

#importing data
data = pd.read_csv('model_dev/dataset_1min_lenti.csv', sep=',')

data_worker = dlm.Dset(data, train_validation=0.3, avg_const=10)
delta_t_avg, q_avg = data_worker.ts_average()
train_validation = data_worker.tv_split(delta_t_avg, q_avg)
delta_t_train = train_validation[0]
q_train = train_validation[1]

#delta_t = np.array(data['delta_t'].values)
delta_t = torch.tensor(delta_t_train).float()
#q = np.array(data['q'].values)
q = torch.tensor(q_train).float()

dl_model = dlm.Model(delta_t, q, bdiv=14, nlayers=32, device=device, dropout=dropout)

#dt_b = dl_model.prep_fwd()

#fwd_output = dl_model.forward(dt_b)

#### training
num_epochs = 50
learning_rate = 0.01
q = q.detach()[0:dl_model.bsize*dl_model.bdiv]
delta_t = delta_t.detach()[0:dl_model.bsize*dl_model.bdiv]

criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(dl_model.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
  for batch in range(dl_model.bdiv):
    optimizer.zero_grad()
    outputs = dl_model.forward(delta_t[batch][0])

    # obtain the loss function
    loss = criterion(outputs, q[batch][0])

    loss.backward()

    optimizer.step()
    if epoch % 1 == 0:
      print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

## prediction part
delta_t_val = torch.tensor(train_validation[3][0:dl_model.bsize*dl_model.bdiv])
q_val = torch.tensor(train_validation[4][0:dl_model.bsize*dl_model.bdiv])

dl_model.eval()

with torch.no_grad():
  prediction = []
  for batch in range(int(len(delta_t)/bsize)):
    prediction = dl_model(delta_t).cpu().detach().numpy()

plt.plot(prediction)
plt.plot(q)
plt.show()

with torch.no_grad():
  prediction = dl_model(delta_t_val).cpu().detach().numpy()

plt.plot(prediction)
plt.plot(q_val)
plt.show()
