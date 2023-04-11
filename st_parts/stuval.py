from datetime import datetime
import sys
sys.path.append('data')
sys.path.append('icons')

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
import os
import pandas as pd
import uval as U

def app():
    col_input1, col_input2, col_x, col_output = st.columns([2.3, 2.3, 0.4, 6])

    with col_input1:
        start_date = st.date_input('Select start date: ')
        start_time = st.text_input('Specify start time (hh:mm):')
        end_date = st.date_input('Select end date:')
        end_time = st.text_input('Specify end time (hh:mm):')
        uploaded_file = st.file_uploader('Upload a file to input data', type='dat')
        if uploaded_file is not None:
            with open('data/raw/' + uploaded_file.name, "wb") as f:
                os.path.relpath('/data/raw')
                f.write(uploaded_file.getbuffer())

        filenames = os.listdir('data/raw/')
        file_name = st.selectbox('Select input file', filenames)
        avg_const = st.number_input('Averaging constant:', value=10, min_value=1)

    with col_input2:
        calc_start_date = st.date_input('Select calculation start date: ')
        calc_start_time = st.text_input('Specify calculation start time (hh:mm):')
        calc_end_date = st.date_input('Select calculation end date:')
        calc_end_time = st.text_input('Specify calculation end time (hh:mm):')
        st.write('Calculation based on selected date/time period')
        calculate_button = st.button('Calculate')
        st.write('Calculation for the optimal date/time period')
        optimize_button = st.button('Optimize')
        swap = st.checkbox('2nd time series')

    if calculate_button:
        start = str(start_date) + ' ' + str(start_time)
        end = str(end_date) + ' ' + str(end_time)
        calc_start = str(calc_start_date) + ' ' + str(calc_start_time)
        calc_end = str(calc_end_date) + ' ' + str(calc_end_time)
        print(start, end, calc_start, calc_end)
        ind_o = U.RawData_index(file_name, start, end)
        first_ind, last_ind = ind_o.dfinit(file_name, start, end)
        first_calc, last_calc = ind_o.dfinit(file_name, calc_start, calc_end)
        calc_rows = last_calc - first_calc
        columns = ind_o.cols(file_name)
        vect_o = U.RawData_series(file_name, first_ind, last_ind, columns)
        ts = vect_o.ex_vect(file_name, first_ind, last_ind, columns)
        ts_u = vect_o.ex_vect(file_name, first_calc, last_calc, columns)
        u_val = U.U_avg(ts, avg_const, ts_u, first_calc, start)
        avg_l = u_val.ts_average(ts, avg_const)
        avg_l_u = u_val.ts_average(ts_u, avg_const)
        avg_l_u = u_val.first_ts_u(avg_l_u, (first_calc-first_ind))
        u_function, u = u_val.u_fun(avg_l_u)
        u_res = u_val.exu(avg_l_u)
        u_plot = U.U_plot(avg_l, avg_l_u, u_function, start, calc_start)
        plot1 = u_plot.u_val_plot(avg_l, avg_l_u, u_function)
        print(u)
    elif optimize_button:
        print('Optimization calculation started!')
        start = (str(start_date) + ' ' + str(start_time))[0:16]
        end = (str(end_date) + ' ' + str(end_time))[0:16]
        calc_start = (str(calc_start_date) + ' ' + str(calc_start_time))[0:16]
        calc_end = (str(calc_end_date) + ' ' + str(calc_end_time))[0:16]
        ind_o = U.RawData_index(file_name, start, end)
        first_ind, last_ind = ind_o.dfinit(file_name, start, end)
        first_calc, last_calc = ind_o.dfinit(file_name, calc_start, calc_end)
        calc_rows = last_calc - first_calc
        columns = ind_o.cols(file_name)
        vect_o = U.RawData_series(file_name, first_ind, last_ind, columns)
        ts = vect_o.ex_vect(file_name, first_ind, last_ind, columns)
        ts_u = vect_o.ex_vect(file_name, first_calc, last_calc, columns)
        u_val = U.U_avg(ts, avg_const, ts_u, first_calc, start)
        avg_l = u_val.ts_average(ts, avg_const)
        if swap:
            qq = avg_l[5]
            print('Calculation for 2nd!')
        else:
            qq = avg_l[4]
            print('Calculaation for 1st!')
        calc_start, calc_end, u_optimal = u_val.u_optim([avg_l[0], avg_l[1], qq, avg_l[6]], start)
        print(calc_start, calc_end, 100 * u_optimal[3:5])
        first_calc, last_calc = ind_o.dfinit(file_name, calc_start, calc_end)
        print(start, end, calc_start, calc_end)
        calc_rows = last_calc - first_calc
        ts_u = vect_o.ex_vect(file_name, first_calc, last_calc, columns)
        u_val = U.U_avg(ts, avg_const, ts_u, first_calc, start)
        avg_l = u_val.ts_average(ts, avg_const)
        avg_l_u = u_val.ts_average(ts_u, avg_const)
        avg_l_u = u_val.first_ts_u(avg_l_u, (first_calc-first_ind))
        u_res = u_val.exu(avg_l_u)
        u_function, u = u_val.u_fun(avg_l_u)
        u_plot = U.U_plot(avg_l, avg_l_u, u_function, start, calc_start)
        plot1 = u_plot.u_val_plot(avg_l, avg_l_u, u_function)
        print(u)
    else:
        u_res = np.zeros((2, 10), dtype=int)
        plot1 = plt.figure()

    with col_output:
        df = pd.DataFrame(u_res, columns=['q (avg)', 'ti (avg)', 'te (avg)', 'U',
                                        'R', 'U24', 'Uf23', 'Ul23', 'dU24', 'dU23'])
        df.index = ['HFM 1', 'HFM 2']
        st.table(df)
        st.pyplot(plot1.figure)

