from logging import Formatter
from os import stat_result
import numpy as np
from numpy.lib.index_tricks import index_exp
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import math

# Finding index from raw data file TRSYS01_Public.dat
class RawData_index:

    def __init__(self, name, start, end):
        self.name = name
        self.start = start
        self.end = end

    # Initial dataframe to find indexes of first and last datapoint
    def dfinit(self, name, start, end):
        df = pd.read_csv('data/raw/' + name,
                    skiprows=1, usecols=['TIMESTAMP'])

        first = -1
        last = -1
        lt = len(start)

        df = df.to_numpy()
        for i in range(len(df)):
            if str(df[i][0])[0:lt] == start:
                first = i
            if str(df[i][0])[0:lt] == end:
                last = i

        if (first == -1) or (last == -1):
            print('ERROR! Timestamp does not work!')
            return

        return first, last

    # finding index of columns
    def cols(self, name):
        df = pd.read_csv('data/raw/' + name, skiprows=1, low_memory=False)

        ti1 = df.columns.get_loc('T11')
        te1 = df.columns.get_loc('T21')
        ti2 = df.columns.get_loc('DT1temp')
        te2 = df.columns.get_loc('DT2temp')
        hf1 = df.columns.get_loc('HF1')
        hf2 = df.columns.get_loc('HF2')

        return [ti1, te1, ti2, te2, hf1, hf2]

# Extracting selected vectors from raw data file TRSYS01_Public.dat
class RawData_series:

    def __init__(self, name, f_ind, l_ind, columns):
        self.name = name
        self.f_ind = f_ind
        self.l_ind = l_ind
        self.columns = columns

    # Extracting function and averaging for every "ts_avg" minutes
    def ex_vect(self, name, f_ind, l_ind, columns):

        first = f_ind + 2
        num_rows = l_ind - f_ind

        x_val = []

        for i in range(num_rows):
            x_val.append(i)

        x_val = np.array(x_val)

        df = pd.read_csv('data/raw/' + name, usecols=columns, skiprows=first,
                        nrows=num_rows, header=None)

        for col in range(len(columns)):
            globals()[f"c{col}"] = df.iloc[:,col].to_numpy()

        return [c0, c1, c2, c3, c4*(-1), c5*(-1), x_val]


class U_avg:

    def __init__(self, ts, avg_const, ts_u, first_calc, start):
        self.ts = ts
        self.avg_const = avg_const
        self.ts_u = ts_u
        self.first_calc = first_calc
        self.start = start

    # First index for timeseries for U-value calculation period
    def first_ts_u(self, ts_u, first_calc):
        if len(ts_u) == 7:
            ts_u_tsp = np.array(ts_u[6]).copy()
            for i in range(len(ts_u_tsp)):
                ts_u_tsp[i] = ts_u_tsp[i] + first_calc
        elif len(ts_u) == 4:
            ts_u_tsp = np.array(ts_u[3]).copy()
            for i in range(len(ts_u_tsp)):
                ts_u_tsp[i] = ts_u_tsp[i] + first_calc
        else:
            print("ERROR! Time series not defined according to the documentation!")
        return ts_u

    # Averaging time series vectors
    def ts_average(self, ts, avg_const):
        c = avg_const
        itr = int(len(ts[0])/c)
        avg_list = []

        for i in range(len(ts)-1):
            globals()[f"ts_help{i}"] = []
            for j in range(itr):
                globals()[f"ts_help{i}"].append(np.sum(ts[i][j*c:(j+1)*c])/c)
            avg_list.append(np.array(globals()[f"ts_help{i}"]))

        x_val = []
        for i in range(itr):
            x_val.append(i*c)
        x_val = np.array(x_val)

        avg_list.append(x_val)

        return np.array(avg_list)

    # U-value function for cases when there are 2 or 1 HFM sensors
    def u_fun(self, ts):
        u_list = []

        if len(ts) == 7:
            ulist1 = []
            ulist2 = []

            tin1_vect = np.array(ts[0]).copy()
            tex1_vect = np.array(ts[1]).copy()
            q1_vect = np.array(ts[4]).copy()
            tin2_vect = np.array(ts[2]).copy()
            tex2_vect = np.array(ts[3]).copy()
            q2_vect = np.array(ts[5]).copy()

            for i in range(len(tin1_vect)):
                n = i + 1

                tin1 = np.sum(tin1_vect[0:n]) / (n + 1)
                tex1 = np.sum(tex1_vect[0:n]) / (n + 1)
                q1 = np.sum(q1_vect[0:n]) / (n + 1)
                u1 = q1 / (tex1 - tin1)

                ulist1.append(u1)

                tin2 = np.sum(tin2_vect[0:n]) / (n + 1)
                tex2 = np.sum(tex2_vect[0:n]) / (n + 1)
                q2 = np.sum(q2_vect[0:n]) / (n + 1)
                u2 = q2 / (tex2 - tin2)

                ulist2.append(u2)

            u = [u1, u2]
            u_list = [ulist1, ulist2]

        elif len(ts) == 4:
            tin_vect = np.array(ts[0]).copy()
            tex_vect = np.array(ts[1]).copy()
            q_vect = np.array(ts[2]).copy()


            for i in range(len(tin_vect)):
                n = i + 1
                tin = np.sum(tin_vect[0:n]) / (n + 1)
                tex = np.sum(tex_vect[0:n]) / (n + 1)
                q = np.sum(q_vect[0:n]) / (n + 1)

                u = q / (tex - tin)

                u_list.append(u)
        else:
           print("ERROR! Time series not defined according to the documentation!")

        return np.array(u_list), u

    # U value, optimizer -- time series [ti, te, q, time]
    #### Old function!
    def u_optim2(self, ts, start_time):

        # averaging time
        tdiff = int(ts[3][1])
        # constraint to min 72 hours
        lag = math.ceil(72 * 60 / tdiff)
        lag_l24 = math.ceil(24 * 60 / tdiff)
        if lag > len(ts[3]):
            print("Measurement ERROR! Measurement did not exceed 72 h.")
        u_store = []

        first = lag - 1
        last = len(ts[3])

        while True:
            first += 1
            if first >= last:
                break

            u = []

            worker = first - 1

            while True:
                worker += 1
                if worker >= last:
                    break

                # U-value for observed period
                ti_vect = np.array(ts[0][first-lag:worker]).copy()
                te_vect = np.array(ts[1][first-lag:worker]).copy()
                q_vect = np.array(ts[2][first-lag:worker]).copy()

                ti = np.sum(ti_vect) / len(ti_vect)
                te = np.sum(te_vect) / len(ti_vect)
                q = np.sum(q_vect) / len(ti_vect)
                up = q / (te - ti)

                # U-value for last 24 h from the end of the observed period
                ti24_vect = ti_vect[0:len(ti_vect) - lag_l24]
                te24_vect = te_vect[0:len(ti_vect) - lag_l24]
                q24_vect = q_vect[0:len(ti_vect) - lag_l24]

                ti24 = np.sum(ti24_vect) / len(ti24_vect)
                te24 = np.sum(te24_vect) / len(te24_vect)
                q24 = np.sum(q24_vect) / len(q24_vect)
                u24 = q24 / (te24 - ti24)

                # relative difference between u and u24
                d24 = abs((up - u24) / up)

                # U value for the first 2/3 of the observed period
                const_23 = int(2 * len(ti_vect) / 3)
                tif23_vect = ti_vect[0:const_23]
                tef23_vect = te_vect[0:const_23]
                qf23_vect = q_vect[0:const_23]

                tif23 = np.sum(tif23_vect) / len(tif23_vect)
                tef23 = np.sum(tef23_vect) / len(tef23_vect)
                qf23 = np.sum(qf23_vect) / len(qf23_vect)
                uf23 = qf23 / (tef23 - tif23)

                # U-value for the last 2/3 of the observed period
                const_l23 = len(ti_vect) - const_23
                til23_vect = ti_vect[const_l23:len(ti_vect)]
                tel23_vect = te_vect[const_l23:len(ti_vect)]
                ql23_vect = q_vect[const_l23:len(ti_vect)]

                til23 = np.sum(til23_vect) / len(til23_vect)
                tel23 = np.sum(tel23_vect) / len(tel23_vect)
                ql23 = np.sum(ql23_vect) / len(ql23_vect)
                ul23 = ql23 / (tel23 - til23)

                d23 = abs((ul23 - uf23) / ul23)

                u.append([first-lag, worker + 1, up, d24, d23, u24, uf23, ul23])

            u = np.array(u)
            ind_min = np.argmin(u[:,3] + u[:,4])
            u_store.append(u[ind_min])

        # list containing following info [first index, last index, u value
        # difference d24, difference d23, u24, uf23, ul23]
        u_store = np.array(u_store)
        ind_min_init = np.argmin(u_store[:,3] + u_store[:,4])
        #print(u_store[ind_min_init])
        #print('dU24 = ', u_store[ind_min_init][3] * 100, ', dUl23 = ', u_store[ind_min_init][4] * 100)
        #print('Ul24, Uf23, Ul23: ', u_store[ind_min_init][5:8], )

        if len(ts) != 4:
            print('ERROR! Time series vector for optimization not defined according to the documentation')
            return
        ind_start = u_store[ind_min_init][0]
        ind_end = u_store[ind_min_init][1]
        start_time = dt.datetime.fromisoformat(self.start)
        optim_start = start_time + dt.timedelta(minutes = (ind_start * self.avg_const))
        if len(u_store) % 2 == 0:
            optim_end = start_time + dt.timedelta(minutes= ((ind_end - 1) * self.avg_const))
        else:
            optim_end = start_time + dt.timedelta(minutes= (ind_end * self.avg_const))
        optim_start_string = optim_start.strftime("%Y-%m-%d %H:%M")
        optim_end_string = optim_end.strftime("%Y-%m-%d %H:%M")

        return optim_start_string, optim_end_string, u_store[ind_min_init]

    # U value, optimizer -- time series [ti, te, q, time]
    def u_optim(self, ts, start_time):

        # averaging time
        tdiff = int(ts[3][1])
        # constraint to min 72 hours
        lag = math.ceil(72 * 60 / tdiff)
        lag_l24 = math.ceil(24 * 60 / tdiff)
        if lag > len(ts[3]):
            print("Measurement ERROR! Measurement did not exceed 72 h.")

        test = 9999
        test2 = 9999

        first = lag - 1
        last = len(ts[3])

        while True:
            first += 1
            if first >= last:
                break

            worker = first - 1

            while True:
                worker += 1
                if worker >= last:
                    break

                # U-value for observed period
                ti_vect = np.array(ts[0][first-lag:worker])
                te_vect = np.array(ts[1][first-lag:worker])
                q_vect = np.array(ts[2][first-lag:worker])

                ti = np.sum(ti_vect) / len(ti_vect)
                te = np.sum(te_vect) / len(ti_vect)
                q = np.sum(q_vect) / len(ti_vect)
                up = q / (te - ti)

                # U-value for last 24 h from the end of the observed period
                ti24_vect = ti_vect[0:len(ti_vect) - lag_l24]
                te24_vect = te_vect[0:len(ti_vect) - lag_l24]
                q24_vect = q_vect[0:len(ti_vect) - lag_l24]

                ti24 = np.sum(ti24_vect) / len(ti24_vect)
                te24 = np.sum(te24_vect) / len(te24_vect)
                q24 = np.sum(q24_vect) / len(q24_vect)
                u24 = q24 / (te24 - ti24)

                # relative difference between u and u24
                d24 = abs((up - u24) / up)

                # U value for the first 2/3 of the observed period
                const_23 = int(2 * len(ti_vect) / 3)
                tif23_vect = ti_vect[0:const_23]
                tef23_vect = te_vect[0:const_23]
                qf23_vect = q_vect[0:const_23]

                tif23 = np.sum(tif23_vect) / len(tif23_vect)
                tef23 = np.sum(tef23_vect) / len(tef23_vect)
                qf23 = np.sum(qf23_vect) / len(qf23_vect)
                uf23 = qf23 / (tef23 - tif23)

                # U-value for the last 2/3 of the observed period
                const_l23 = len(ti_vect) - const_23
                til23_vect = ti_vect[const_l23:len(ti_vect)]
                tel23_vect = te_vect[const_l23:len(ti_vect)]
                ql23_vect = q_vect[const_l23:len(ti_vect)]

                til23 = np.sum(til23_vect) / len(til23_vect)
                tel23 = np.sum(tel23_vect) / len(tel23_vect)
                ql23 = np.sum(ql23_vect) / len(ql23_vect)
                ul23 = ql23 / (tel23 - til23)

                d23 = abs((ul23 - uf23) / ul23)

                if (d23 + d24) < test:
                    test = d23 + d24
                    ind_min_loop = first-lag
                    ind_end_loop = worker + 1
                    up_loop = up
                    d24_loop = d24
                    d23_loop = d23
                    u24_loop = u24
                    uf23_loop = uf23
                    ul23_loop = ul23

            if (d24_loop + d23_loop) < test2:
                test2 = d23_loop + d24_loop
                u_store = [ind_min_loop, ind_end_loop, up_loop, d24_loop,
                    d23_loop, u24_loop, uf23_loop, ul23_loop]

        if len(ts) != 4:
            print('ERROR! Time series vector for optimization not defined according to the documentation')
            return

        #  u_store -- list containing following info [first index, last index, u value
        # difference d24, difference d23, u24, uf23, ul23]
        ind_start = u_store[0]
        ind_end = u_store[1]
        start_time = dt.datetime.fromisoformat(self.start)
        optim_start = start_time + dt.timedelta(minutes = (ind_start * self.avg_const))
        if len(u_store) % 2 == 0:
            optim_end = start_time + dt.timedelta(minutes= ((ind_end - 1) * self.avg_const))
        else:
            optim_end = start_time + dt.timedelta(minutes= (ind_end * self.avg_const))
        optim_start_string = optim_start.strftime("%Y-%m-%d %H:%M")
        optim_end_string = optim_end.strftime("%Y-%m-%d %H:%M")

        return optim_start_string, optim_end_string, u_store

    # Extraction of standard values
    def exu(self, lu):
        lag_l24 = math.ceil(24 * 60 / self.avg_const)

        if len(lu) == 7:
            ti1 = np.array(lu[0]).copy()
            te1 = np.array(lu[1]).copy()
            q1 = np.array(lu[4]).copy()
            ti2 = np.array(lu[2]).copy()
            te2 = np.array(lu[3]).copy()
            q2 = np.array(lu[5]).copy()
            t1i_avg = np.sum(ti1) / len(ti1)
            t1e_avg = np.sum(te1) / len(te1)
            q1_avg = np.sum(q1) / len(q1)
            t2i_avg = np.sum(ti2) / len(ti2)
            t2e_avg = np.sum(te2) / len(te2)
            q2_avg = np.sum(q2) / len(q2)
            U1 = q1_avg / (t1e_avg - t1i_avg)
            U2 = q2_avg / (t2e_avg - t2i_avg)
            R1 = 1 / U1
            R2 = 1 / U2

            ti1_24_avg = np.sum(ti1[0:len(ti1) - lag_l24]) / len(ti1[0:len(ti1) - lag_l24])
            te1_24_avg = np.sum(te1[0:len(te1) - lag_l24]) / len(ti1[0:len(ti1) - lag_l24])
            q1_24_avg = np.sum(q1[0:len(q1) - lag_l24]) / len(ti1[0:len(ti1) - lag_l24])
            ti2_24_avg = np.sum(ti2[0:len(ti2) - lag_l24]) / len(ti1[0:len(ti1) - lag_l24])
            te2_24_avg = np.sum(te2[0:len(te2) - lag_l24]) / len(ti1[0:len(ti1) - lag_l24])
            q2_24_avg = np.sum(q2[0:len(q2) - lag_l24]) / len(ti1[0:len(ti1) - lag_l24])

            U1_24 = q1_24_avg / (te1_24_avg - ti1_24_avg)
            U2_24 = q2_24_avg / (te2_24_avg - ti2_24_avg)
            dU1_24 = abs((U1 - U1_24) / U1 * 100)
            dU2_24 = abs((U2 - U2_24) / U2 * 100)

            const_23 = int(2 * len(ti1) / 3)
            tif1_23_avg = np.sum(ti1[0:const_23]) / len(ti1[0:const_23])
            tef1_23_avg = np.sum(te1[0:const_23]) / len(te1[0:const_23])
            qf1_23_avg = np.sum(q1[0:const_23]) / len(q1[0:const_23])
            tif2_23_avg = np.sum(ti2[0:const_23]) / len(ti2[0:const_23])
            tef2_23_avg = np.sum(te2[0:const_23]) / len(te2[0:const_23])
            qf2_23_avg = np.sum(q2[0:const_23]) / len(q2[0:const_23])
            U1_f23 = qf1_23_avg / (tef1_23_avg - tif1_23_avg)
            U2_f23 = qf2_23_avg / (tef2_23_avg - tif2_23_avg)

            const_l23 = len(ti1) - const_23
            til1_23_avg = np.sum(ti1[const_l23:len(ti1)]) / len(ti1[const_l23:len(ti1)])
            tel1_23_avg = np.sum(te1[const_l23:len(te1)]) / len(te1[const_l23:len(te1)])
            ql1_23_avg = np.sum(q1[const_l23:len(ti1)]) / len(ti1[const_l23:len(ti1)])
            til2_23_avg = np.sum(ti2[const_l23:len(ti1)]) / len(ti1[const_l23:len(ti1)])
            tel2_23_avg = np.sum(te2[const_l23:len(ti1)]) / len(ti1[const_l23:len(ti1)])
            ql2_23_avg = np.sum(q2[const_l23:len(ti1)]) / len(ti1[const_l23:len(ti1)])
            U1_l23 = ql1_23_avg / (tel1_23_avg - til1_23_avg)
            U2_l23 = ql2_23_avg / (tel2_23_avg - til2_23_avg)

            dU1_23 = abs((U1_l23 - U1_f23) / U1_l23 * 100)
            dU2_23 = abs((U2_l23 - U2_f23) / U2_l23 * 100)

            u = [[q1_avg, t1i_avg, t1e_avg, U1, R1, U1_24, U1_f23, U1_l23, dU1_24, dU1_23],
                 [q2_avg, t2i_avg, t2e_avg, U2, R2, U2_24, U2_f23, U2_l23, dU2_24, dU2_23]]
            return u

        elif len(lu) == 4:
            ti1 = np.array(lu[0]).copy()
            te1 = np.array(lu[1]).copy()
            q1 = np.array(lu[4]).copy()

            t1i_avg = np.sum(ti1) / len(ti1)
            t1e_avg = np.sum(te1) / len(te1)
            q1_avg = np.sum(q1) / len(q1)
            U1 = q1_avg / (t1e_avg - t1i_avg)
            R1 = 1 / U1

            ti1_24_avg = np.sum(ti1[0:len(ti1) - lag_l24]) / len(ti1[0:len(ti1) - lag_l24])
            te1_24_avg = np.sum(te1[0:len(te1) - lag_l24]) / len(ti1[0:len(ti1) - lag_l24])
            q1_24_avg = np.sum(q1[0:len(q1) - lag_l24]) / len(ti1[0:len(ti1) - lag_l24])
            U1_24 = q1_24_avg / (te1_24_avg - ti1_24_avg)
            dU1_24 = abs((U1 - U1_24) / U1 * 100)

            const_23 = int(2 * len(ti1) / 3)
            tif1_23_avg = np.sum(ti1[0:const_23]) / len(ti1[0:const_23])
            tef1_23_avg = np.sum(te1[0:const_23]) / len(te1[0:const_23])
            qf1_23_avg = np.sum(q1[0:const_23]) / len(q1[0:const_23])
            U1_f23 = qf1_23_avg / (tef1_23_avg - tif1_23_avg)

            const_l23 = len(ti1) - const_23
            til1_23_avg = np.sum(ti1[const_l23:len(ti1)]) / len(ti1[const_l23:len(ti1)])
            tel1_23_avg = np.sum(te1[const_l23:len(te1)]) / len(te1[const_l23:len(te1)])
            ql1_23_avg = np.sum(q1[const_l23:len(ti1)]) / len(ti1[const_l23:len(ti1)])
            U1_l23 = ql1_23_avg / (tel1_23_avg - til1_23_avg)

            dU1_23 = abs((U1_l23 - U1_f23) / U1_l23 * 100)

            u = [q1_avg, t1i_avg, t1e_avg, U1, R1, U1_24, U1_f23, U1_l23, dU1_24, dU1_23]
            return u

        else:
            print("ERROR! Time series not defined according to the documentation!")
            return



class U_plot:

    def __init__(self, ts, ts_u, u_fun, start, calc_start):
        self.ts = ts
        self.ts_u = ts_u
        self.u_fun = u_fun
        self.start = start
        self.calc_start = calc_start

        self.start = dt.datetime.fromisoformat(start)
        self.calc_start = dt.datetime.fromisoformat(calc_start)

    def u_val_plot(self, ts, ts_u, u_fun):
        if len(ts) == 7:

            # x_axis datetime
            date_list = []
            for i in range(len(ts[0])):
                date_list.append(self.start + dt.timedelta(minutes=ts[6][i]))

            # x_axis datetime calculation period
            date_list_u = []
            for i in range(len(ts_u[0])):
                date_list_u.append(self.calc_start + dt.timedelta(minutes=ts[6][i]))

            fig1, ax1 = plt.subplots(figsize=(10.16, 6.28))
            ax1.set_title('HFM results - two sensors')
            ax1.set_xlabel('Time [date/time]')
            ax1.set_ylabel('Temperature [°C]')
            ax1.plot(date_list, ts[0], color='orangered')
            ax1.plot(date_list, ts[1], color='navy')

            ax2 = ax1.twinx()
            ax2.set_ylabel('Heat flux [$W / m^2$]')
            ax2.plot(date_list, ts[4], color='forestgreen')
            ax2.plot(date_list_u, u_fun[0], '--', color='darkmagenta')

            ax1.plot(date_list, ts[2], color='darkorange')
            ax1.plot(date_list, ts[3], color='darkcyan')

            ax2.plot(date_list, ts[5], color='yellowgreen')
            ax2.plot(date_list_u, u_fun[1],'--', color='hotpink')

            box = ax1.get_position()
            ax1.set_position([box.x0, box.y0 + box.height * 0.2,
                              box.width, box.height * 0.8])
            ax1.legend(['$T_{int,1}$','$T_{ext,1}$', '$T_{int,2}$',
            '$T_{ext,2}$'], loc='lower center',
            bbox_to_anchor=(0.5, -0.3), fancybox=True, ncol=5)
            ax2.legend(['$HF_1$', '$U_1$', '$HF_2$', '$U_2$'], loc='lower center',
                       bbox_to_anchor=(0.5, -0.4), fancybox=True, ncol=5)

            y_min, y_max = ax1.get_ylim()
            ax1.plot([date_list_u[0], date_list_u[0]], [y_min, y_max], ':',
            color='cadetblue')
            ax1.plot([date_list_u[-1], date_list_u[-1]], [y_min, y_max], ':',
            color='cadetblue')

            return ax1

        elif len(ts) == 4:

            # x_axis datetime
            date_list = []
            for i in range(len(ts[0])):
                date_list.append(self.start + dt.timedelta(minutes=ts[3][i]))

            # x_axis datetime calculation period
            date_list_u = []
            for i in range(len(ts_u[0])):
                date_list_u.append(self.calc_start + dt.timedelta(minutes=ts[3][i]))

            fig1, ax1 = plt.subplots(figsize=(10.16, 6.28))
            ax1.set_title('HFM results')
            ax1.set_xlabel('Time [date/time]')
            ax1.set_ylabel('Temperature [°C]')
            ax1.plot(date_list, ts[0], color='orangered')
            ax1.plot(date_list, ts[1], color='navy')

            ax2 = ax1.twinx()
            ax2.set_ylabel('Heat flux [$W / m^2$]')
            ax2.plot(date_list, ts[2], color='forestgreen')
            ax2.plot(date_list_u, u_fun, '--', color='darkmagenta')

            box = ax1.get_position()
            ax1.set_position([box.x0, box.y0 + box.height * 0.2,
                              box.width, box.height * 0.8])
            ax1.legend(['$T_{int,1}$', '$T_{ext,1}$'], loc='lower center',
            bbox_to_anchor=(0.5, -0.3), fancybox=True, ncol=5)
            ax2.legend(['$HF_1$', '$U_1$'], loc='lower center',
                       bbox_to_anchor=(0.5, -0.4), fancybox=True, ncol=5)

            y_min, y_max = ax1.get_ylim()
            ax1.plot([date_list_u[0], date_list_u[0]], [y_min, y_max], ':',
                     color='cadetblue')
            ax1.plot([date_list_u[-1], date_list_u[-1]], [y_min, y_max], ':',
                     color='cadetblue')

            return ax1

        else:
            print("Error! Time series not defined according to the documentation")
            return
