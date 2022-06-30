import sys

sys.path.append('data')
sys.path.append('icons')
sys.path.append('dl_models')

import os
import csv
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import streamlit as st
import torch
import time
import regression as worker
import scipy

cm = 1/2.54

def app():
    torch.manual_seed(1618)
    uploaded_file = st.sidebar.file_uploader('Upload a file to input data', type='csv')
    if uploaded_file is not None:
        with open('data/model_input/' + uploaded_file.name, 'wb') as f:
            os.path.realpath('/data/model_input')
            f.write(uploaded_file.getbuffer())
    filenames = os.listdir('data/model_input')
    file_name = st.sidebar.selectbox('Select input file', filenames)
    train_validation = st.sidebar.number_input('Specify train/valdiation ratio:', value=0.75)
    bsize = st.sidebar.number_input('Specify batch size', value=128)
    nlayers = st.sidebar.number_input('Specify number of LSTM layers', min_value=0, value=8)
    device = st.sidebar.selectbox('Specify execution device', ['cuda', 'cpu'])
    dropout = st.sidebar.number_input('Specify dropout probability', min_value=0.00, max_value=1.00, value=0.50)
    select_l2l = st.sidebar.number_input('Select lambda for l2 reg', value=0.0005, min_value=0.0000)
    nepochs = st.sidebar.number_input('Specify number of epochs', min_value=1, value=50)
    model_name = st.sidebar.text_input('Specify model name', value=file_name[0:-4]+'_tv' + str(train_validation)[
                                       2:]+'_b'+str(bsize)+'_nl'+str(nlayers)+'_pd'+str(dropout)[2:]+'_l2l'+str(select_l2l)[2:])
    avg_const = 1 #st.sidebar.number_input('Specify average constant:', min_value=1, max_value=60, value=1)
    expected_u = st.sidebar.number_input('Specify expected U-value', value=4.00)
    model_save = st.sidebar.checkbox('Model save', value=False)
    run_button = st.sidebar.button('Run')

    if run_button:
        ts = pd.read_csv('data/model_input/' + file_name, sep=',')
        data_worker = worker.Dset(ts, train_validation=train_validation, avg_const=avg_const)
        v1_avg, v2_avg = data_worker.ts_average()
        split_list = data_worker.tv_split(v1_avg, v2_avg, bsize)
        v1_train = torch.tensor(split_list[0])
        v2_train = torch.tensor(split_list[1])

        if bsize > len(v1_train):
            st.warning('Batch size is greater than the vector length! bsize = len(vector)')

        model = worker.Dl_model(bsize=bsize, nlayers=nlayers, device=device, dropout=dropout)#worker.MLP(bsize=bsize, nperceptrons=3, device=device, dropout=dropout)#
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = torch.nn.MSELoss()
        bdiv = int(len(v1_train) / bsize)
        regulizer = v2_train[0:bsize*bdiv].to(device).float().view(bdiv, 1, bsize)
        v1_train = v1_train[0:bsize*bdiv].to(device).float().view(bdiv, 1, bsize)
        x1 = split_list[2][0:bdiv*bsize]
        v1_validation = np.array(split_list[3][0:len(split_list[3])])
        bdiv_validation = int(len(v1_validation)/bsize)
        v1_validation = v1_validation[0:bdiv_validation*bsize]
        v2_validation = np.array(split_list[4][0:bdiv_validation*bsize])
        x2 = split_list[5][0:bdiv_validation*bsize]

        loss_list = []
        loss_val_list = []

        info = st.empty()

        for epoch in range(nepochs):
            loss_epoch = []
            start_time = time.time()
            for batch in range(bdiv):
                model.train()
                #start_time = time.time()
                optimizer.zero_grad()
                pred = model.forward(v1_train[batch][0])
                if regulizer.device != pred.device:
                    regulizer = regulizer.to('cpu').float()
                    regulizer = regulizer.flatten()
                    st.warning('WARRNING! Device = "cpu"!')
                loss = loss_fn(pred, regulizer[batch][0])
                l2_reg = torch.tensor(0.).to(device)
                for param in model.parameters():
                    l2_reg += torch.norm(param).to(device)
                loss += select_l2l * l2_reg
                loss.backward()
                optimizer.step()
                loss_epoch.append(loss.item())
            loss_list.append(loss_epoch[-1])

            with torch.no_grad():
                prediction_train = []
                for batch in range(bdiv):
                    model.eval()
                    prediction = model.forward(v1_train[batch][0])
                    prediction = torch.flatten(prediction).cpu().detach().numpy()
                    prediction_train = np.append(prediction_train, prediction)

                prediction_validation = []
                v1_validation = torch.tensor(v1_validation).to(device).view(bdiv_validation, 1, bsize)
                for batch in range(bdiv_validation):
                    model.eval()
                    prediction = model.forward(v1_validation[batch][0])
                    prediction = torch.flatten(prediction).cpu().detach().numpy()
                    prediction_validation = np.append(prediction_validation, prediction)
                loss_val = (np.square(prediction_validation - v2_validation)).mean()
                loss_val_list.append(loss_val)
            info.warning(f'Epoch: {epoch + 1}, train loss: {loss.item()}, validation loss: {loss_val.item()} execution per epoch: {time.time() - start_time}')

        bdiv_all = int((len(v2_train.flatten()) + len(v2_validation.flatten())) / bsize)
        v1_comb = torch.cat((v1_train.flatten(), v1_validation.flatten())).view(bdiv_all, 1, bsize)
        v2_comb = torch.cat((v2_train.flatten(), torch.tensor(v2_validation))).view(bdiv_all, 1, bsize)

        q_pred = np.array([])
        with torch.no_grad():
            for batch in range(bdiv_all):
                model.eval()
                q_prediction = model.forward(v1_comb[batch][0])
                q_prediction = torch.flatten(q_prediction).cpu().detach().numpy()
                q_pred = np.append(q_pred, q_prediction)

        q_heading = ['q_pred_' + model_name]
        with open('results/q_pred/' + model_name + '.csv', 'w') as file:
            write = csv.writer(file)
            write.writerow(q_heading)
            write.writerows([q_pred[i]] for i in range(len(q_pred)))

        u_m, U_m = worker.u_val(v1_comb.cpu().flatten().numpy(), v2_comb.cpu().flatten().numpy())
        u_p, U_p = worker.u_val(v1_comb.cpu().flatten().numpy(), q_pred)
        rel_errorp = round(abs((expected_u - U_p) / expected_u) * 100, 2)
        rel_errorm = round(abs((expected_u - U_m) / expected_u) * 100, 2)

        fig = plt.figure(figsize=(9, 9))
        #fig.suptitle(f'{model_name}, greška: {"{0:.2f}".format(abs_error)} %')
        G = gridspec.GridSpec(7, 7)

        #### HFM input and prediction ###
        axes_1 = fig.add_subplot(G[0:3,:])
        #axes_1.set_title('Ulazni podaci')
        axes_1.set_xlabel('Vrijeme [minute]')
        axes_1.set_ylabel('Razlika temperature [°C]')
        axes_2 = axes_1.twinx()
        axes_2.set_ylabel('Toplinski tok [$W / m^2$]')
        axes_1.scatter(split_list[2], split_list[0],
                       s=0.2, color=(0.5, 0.2, 0.5))
        axes_2.scatter(split_list[2], split_list[1],
                       s=0.2, color=(0.1, 0.2, 0.5))
        axes_1.scatter(split_list[5], split_list[3],
                       s=0.2, color=(0.5, 0.4, 0.5, 0.4))
        axes_2.scatter(split_list[5], split_list[4],
                       s=0.2, color=(0.1, 0.4, 0.5, 0.4))
        axes_2.plot(q_pred, color=(0.1, 0.2, 0.7, 0.6))
        box = axes_1.get_position()
        axes_1.set_position([box.x0, box.y0 + box.height * 0.2,
                             box.width, box.height * 0.8])
        axes_1.legend(['$\Delta T$ trening', '$\Delta T$ validacija'], loc='lower center', markerscale=5,
                      bbox_to_anchor=(0.25, -0.5), fancybox=True, fontsize=9)
        axes_2.legend(['q predikcija', 'q trening', 'q validacija'], loc='lower center', markerscale=5,
                      bbox_to_anchor=(0.75, -0.5), fancybox=True, fontsize=9)

        ### Training and validation losses
        axes_3 = fig.add_subplot(G[4:6,0:3])
        #axes_3.set_title('Gubitak treninga i validacije')
        axes_3.set_xlabel('Epoha')
        axes_3.set_ylabel('RMSE gubitak')
        loss_list = np.sqrt(loss_list)
        loss_val_list = np.sqrt(loss_val_list)
        axes_3.plot(loss_list, color=(0.5, 0.4, 0.5, 0.6))
        axes_3.plot(loss_val_list, color=(0.1, 0.4, 0.5, 0.6))
        box = axes_3.get_position()
        axes_3.set_position([box.x0, box.y0 + box.height * 0.2,
                             box.width, box.height * 0.8])
        axes_3.legend(['Gubitak treninga', 'Gubitak validacije'], loc='lower center',
                      bbox_to_anchor=(0.67, 0.6), fancybox=True, fontsize=9)
        ### Predicted v.s. measured
        axes_4 = fig.add_subplot(G[4:6,4:7])
        #axes_4.set_title('Odnos predikcija / izmjereno')
        axes_4.set_xlabel('Toplinski tok - izmjereno')
        axes_4.set_ylabel('Toplinski tok - predikcija')
        #axes_4.scatter(v2_comb.flatten(), q_pred, s=0.05, color=(0.1, 0.4, 0.5, 0.2))
        q_msum, q_psum = worker.q_sum(v2_comb.flatten().numpy(), q_pred)
        min_q = int(np.min([q_msum, q_psum]))
        max_q = int(np.max([q_msum, q_psum]))
        axes_4.scatter(q_msum, q_psum, s=0.2, color=(0.1, 0.4, 0.5, 0.1))
        axes_4.plot([i for i in range(min_q, max_q)],
                    [i for i in range(min_q, max_q)], linestyle='--')
        axes_4.tick_params(axis='x', labelrotation=33)
        print(f'linregress: {scipy.stats.linregress(q_psum, q_msum)}')
        rsq_k1 = worker.rsq_k1(q_psum, q_msum)
        print(f'Rsquared for k=1: {rsq_k1}')
        box = axes_4.get_position()
        axes_4.set_position([box.x0, box.y0 + box.height * 0.2,
                             box.width, box.height * 0.8])
        axes_4.legend([f'R$^2$ (k=1) = {format(rsq_k1, ".4f")}'],
                      bbox_to_anchor=(0.63, 0.95), fancybox=True, fontsize=9)

        col1, col2, col3 = st.columns([0.5, 3, 0.5])
        with col2:
            st.pyplot(fig)

        fig = plt.figure(figsize=(7, 7))
        G = gridspec.GridSpec(7, 1)
        ax1 = fig.add_subplot(G[0:3, 0])
        ax1.set_xlabel('Vrijeme [minute]')
        ax1.set_ylabel('U-vrijednost [W/($m^2K$)]')
        ax2 = ax1.twinx()
        ax2.set_ylabel('Ukupan toplinski tok [$W / m^2$]')
        if expected_u != -666:
            min_ = int(expected_u) - 3
            if min_ < 0:
                min_ = 0
            ax1.set_ylim([min_, int(expected_u) + 3])
        else:
            u_mean = (u_m[-1] + u_p[-1]) / 2
            min_ = u_mean - 1
            max_ = u_mean + 1
            ax1.set_ylim([0, max_])
        ax1.plot(u_m, color=(0.1, 0.2, 0.5, 0.4))
        ax1.plot(u_p, color=(0.5, 0.2, 0.5, 0.4))
        if expected_u != -666:
            ax1.plot([i for i in range(len(u_m))],
                     [expected_u for i in range(len(u_m))], color=(0.1, 0.1, 0.1, 0.4),
                     linestyle='--')
        ax2.plot(q_msum, color=(0.1, 0.2, 0.5))
        ax2.plot(q_psum, color=(0.5, 0.2, 0.5))
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0 + box.height * 0.2,
                          box.width, box.height * 0.8])
        ax1.legend(['U-vrijednost (mjerenje)', 'U-vrijednost (procjena)', 'U-vrijednost (očekivana)'],
                   loc='lower center', markerscale=5, bbox_to_anchor=(0.15, -0.7), fancybox=True, fontsize=9)
        ax2.legend(['Toplinski tok (mjerenje)', 'Toplinski tok (procjena)'], loc='lower center', markerscale=5,
                   bbox_to_anchor=(0.85, -0.7), fancybox=True, fontsize=9)
        if expected_u != -666:
            ax3 = fig.add_subplot(G[4:7, 0])
            ax3.set_xlabel('Vrijeme [minute]')
            ax3.set_ylabel('Relativna pogreška [%]')
            ax3.set_ylim([-55, 55])
            ax3.yaxis.set_ticks([i for i in range(-50, 60, 10)])
            u_iso = [expected_u for i in range(len(u_m))]
            rel_err = worker.rel_err(u_iso, u_p)
            rel_err2 = worker.rel_err(u_iso, u_m)
            ax3.plot(rel_err, color=(0.5, 0.2, 0.5, 0.4))
            ax3.plot(rel_err2, color=(0.1, 0.2, 0.5, 0.4))
            ax3.legend(['$U_{DL}$ relativna pogreška', '$U_{HFM}$ relativna pogreška'],
                    loc='lower center', markerscale=5, fancybox=True, fontsize=9)
            ax3.plot([i for i in range(len(u_m))],
                    [-10 for i in range(len(u_m))], linestyle=(0,(1,1)), color='green')
            ax3.plot([i for i in range(len(u_m))],
                    [10 for i in range(len(u_m))], linestyle=(0,(1,1)), color='green')
            ax3.plot([i for i in range(len(u_m))],
                    [-20 for i in range(len(u_m))], linestyle=(0,(5,5)), color='royalblue')
            ax3.plot([i for i in range(len(u_m))],
                    [20 for i in range(len(u_m))], linestyle=(0,(5,5)), color='royalblue')
            ax3.plot([i for i in range(len(u_m))],
                    [0 for i in range(len(u_m))], color='gray')

        #plt.savefig('results/plots/' + model_name + '.png', dpi=300)
        col1, col2, col3 = st.columns([0.5, 3, 0.5])
        with col2:
            st.pyplot(fig)

        print(f'Um = {U_m}, Up = {U_p}, Uiso = {expected_u}, ErrP = {rel_errorp}, ErrM = {rel_errorm}')

        # Saving the model
        if model_save:
            model_name = 'results/models/' + model_name + '.pt'
            torch.save(model, model_name)

## pogledaj warning za torch tensor!
## Popravi da Run uvijek radi
