import sys
sys.path.append('dl_models')
sys.path.append('data')

import numpy as np
import regression
import uval as U
import matplotlib.pyplot as plt

######## Input ########
file_name = 'TRSYS01_Public-komora1.dat'

start = '2021-07-02 13:00'
end = '2021-12-27 09:00'

avg_const = 30

train_validation = 1

loss_function = 'MSE'
optimizer = 'Adam'
lr = 0.01
batch_size = 3
epochs = 1000
########################

ind_o = U.RawData_index(file_name, start, end)
first_ind, last_ind = ind_o.dfinit(file_name, start, end)
columns = ind_o.cols(file_name)
vect_o = U.RawData_series(file_name, first_ind, last_ind, columns)

# List 'ts' contains following vectors
# T11, T21, DT1temp, DT2temp, HF1, HF2, TIMESTAMP
ts = vect_o.ex_vect(file_name, first_ind, last_ind, columns)

# Extraction of results for sensor set 1 (sset = 1)
pre = regression.Dset(ts, avg_const, train_validation)
ts_avg = pre.ts_average()

# dt and q are vectors time difference and heat flux extracted from ts
dt, q = pre.ts_ex(ts_avg)
#dt, q = regression.ts_ex(ts, sset = 2)

dt_train, q_train, dt_validation, q_validation = pre.tv_split(dt, q)

plt.plot([i for i in range(len(dt_train))], dt_train)
plt.plot([i for i in range(len(dt_train))], q_train)
plt.plot([i for i in range(len(dt_train),len(dt))], dt_validation)
plt.plot([i for i in range(len(dt_train),len(dt))], q_validation)
plt.show()

