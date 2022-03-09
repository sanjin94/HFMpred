import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import model_classes as dlm
bsize = 48

model = torch.load('model_dev/test/model/32lstm.pt')
model.eval()

data_val = pd.read_csv('model_dev/test/dataset_1min_izolacija.csv', sep=',')
data_worker_val = dlm.Dset(data_val, train_validation=1, avg_const=1)
delta_t_avg, q_avg = data_worker_val.ts_average()
train_validation = data_worker_val.tv_split(delta_t_avg, q_avg)
bdiv = int(len(train_validation[0]) / bsize)
delta_t_validation = torch.tensor(train_validation[0][0:bdiv*bsize]).view(bdiv, 1, bsize).float()
q_validation = train_validation[1]

with torch.no_grad():
  prediction = []
  for batch in range(bdiv):
    prediction_validation = model(delta_t_validation[batch]).cpu().detach().numpy()
    prediction = np.append(prediction, prediction_validation)

plt.plot(prediction)
plt.plot(q_validation)
plt.show()

