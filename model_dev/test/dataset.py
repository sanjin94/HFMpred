import sys
sys.path.append('data')

import matplotlib.pyplot as plt
import uval as U
import numpy as np
import csv

file_name = 'rezultati_210222.dat'#

start = '2021-12-15 09:00'
end = '2022-01-25 09:00'
calc_start = '2021-12-15 09:00'
calc_end = '2022-01-25 09:00'

avg_const = 1

ind_o = U.RawData_index(file_name, start, end)
first_ind, last_ind = ind_o.dfinit(file_name, start, end)
first_calc, last_calc = ind_o.dfinit(file_name, calc_start, calc_end)
first_calc = first_calc
calc_rows = last_calc - first_calc
columns = ind_o.cols(file_name)
vect_o = U.RawData_series(file_name, first_ind, last_ind, columns)
# List "ts" contains following vectors
# T11, T21, DT1temp, DT2temp, HF1, HF2, TIMESTAMP
ts = vect_o.ex_vect(file_name, first_ind, last_ind, columns)
ts_u = vect_o.ex_vect(file_name, first_calc, last_calc, columns)
# U-value operation object

u_val = U.U_avg(ts, avg_const, ts_u, first_calc, start)
# average function
avg_l = u_val.ts_average(ts, avg_const)

delta_t = np.array(avg_l[0] - avg_l[1]).copy()
q = np.array(avg_l[4]).copy()

### Plot dataset
fig1, ax1 = plt.subplots(figsize=(10.16, 6.28))
ax1.set_title('HFMpred training dataset')
ax1.set_xlabel('Time [x 10 min]')
ax1.set_ylabel('Temperature difference [°C]')
ax1.plot(delta_t, color='navy')
ax2 = ax1.twinx()
ax2.set_ylabel('Heat flux [$W / m^2$]')
ax2.plot(q, color='orangered')
box = ax1.get_position()
ax1.set_position([box.x0, box.y0 + box.height * 0.2,
                  box.width, box.height * 0.8])
ax1.legend(['$\Delta$ t'], loc='lower center', bbox_to_anchor=(0.5, -0.3), fancybox=True)
ax2.legend(['Heat flux'], loc='lower center',
           bbox_to_anchor=(0.5, -0.4), fancybox=True)
plt.show()

### CSV saving of the list of vectors delta_T and q
heading = ['delta_t', 'q']

with open('model_dev/dataset_1min_izolacija.csv', 'w') as file:
   write = csv.writer(file)
   write.writerow(heading)
   write.writerows([delta_t[i], q[i]] for i in range(len(q)))
