import sys
sys.path.append('data')

import matplotlib.pyplot as plt
import uval as U
import numpy as np

file_name = 'TRSYS01_Public-komora1.dat'#

start = '2021-07-02 13:00'  # '2021-11-16 17:30'  #
end = '2021-12-27 09:00'  # '2021-11-30 16:30'  #
calc_start = '2021-07-02 13:00'  # '2021-11-16 17:30'  #
calc_end = '2021-12-27 09:00'  # '2021-11-30 16:30'  #

avg_const = 30

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
avg_l_u = u_val.ts_average(ts_u, avg_const)

# Index change for U-value calculation
avg_l_u = u_val.first_ts_u(avg_l_u, (first_calc-first_ind))

# defining u_val timeseries and last timestep U-value
u_function, u = u_val.u_fun(avg_l_u)

# ploting the results
u_plot = U.U_plot(avg_l, avg_l_u, u_function, start, calc_start)
plot1 = u_plot.u_val_plot(avg_l, avg_l_u, u_function)

plt.show()

#calc_start, calc_end, u_optimal = u_val.u_optim([avg_l[0], avg_l[1], avg_l[4], avg_l[6]], start)

#print(calc_start, calc_end, u_optimal)

