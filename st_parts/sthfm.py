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
import streamlit as st
import torch
import time
import regression as worker

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
    select_l2l = st.sidebar.number_input('Select lambda for l2 reg', value=0.005, min_value=0.0001)
    nepochs = st.sidebar.number_input('Specify number of epochs', min_value=1, value=50)
    model_name = st.sidebar.text_input('Specify model name', value=file_name[0:-4]+'_tv' + str(train_validation)[
                                       2:]+'_b'+str(bsize)+'_nl'+str(nlayers)+'_pd'+str(dropout)[2:]+'_l2l'+str(select_l2l)[2:])
    avg_const = 1 #st.sidebar.number_input('Specify average constant:', min_value=1, max_value=60, value=1)
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
                loss = loss_fn(pred, regulizer[batch][0])
                if regulizer.device != pred.device:
                    regulizer = regulizer.to('cpu').float()
                    regulizer = regulizer.flatten()
                    st.warning('WARRNING! Device = "cpu"!')
                l2_lambda = 0
                l2_reg = torch.tensor(0.).to(device)
                for param in model.parameters():
                    l2_reg += torch.norm(param).to(device)
                loss += l2_lambda * l2_reg
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
            info.warning(f'Epoch: {epoch}, train loss: {loss.item()}, validation loss: {loss_val.item()} execution per epoch: {time.time() - start_time}')

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
        abs_error = round(abs((U_m - U_p) / U_p) * 100, 2)

        fig, axs = plt.subplots(3, figsize=(8, 13))
        #fig.set_size_inches(8, 13)
        #gs = fig.add_gridspec(5, hspace=0)
        #axs = gs.subplots(figsize=(13, 8))
        # izbaci kasnije
        fig.suptitle(f'{model_name}, greška: {"{0:.2f}".format(abs_error)} %')
        axs[0].set_title('Ulazni podaci')
        axs[0].set_xlabel('Vrijeme [minute]')
        axs[0].set_ylabel('Razlika temperature [°C]')
        ax2 = axs[0].twinx()
        ax2.set_ylabel('Toplinski tok [$W / m^2$]')
        axs[0].scatter(split_list[2], split_list[0], s=0.2, color=(0.5, 0.2, 0.5))
        ax2.scatter(split_list[2], split_list[1], s=0.2, color=(0.1, 0.2, 0.5))
        axs[0].scatter(split_list[5], split_list[3], s=0.2, color=(0.5, 0.4, 0.5, 0.4))
        ax2.scatter(split_list[5], split_list[4], s=0.2, color=(0.1, 0.4, 0.5, 0.4))
        box = axs[0].get_position()
        axs[0].set_position([box.x0, box.y0 + box.height * 0.2,
                          box.width, box.height * 0.8])
        axs[0].legend(['$\Delta T$ trening', '$\Delta T$ validacija'], loc='lower center', markerscale=5,
                  bbox_to_anchor=(0.25, -0.35), fancybox=True)
        ax2.legend(['q trening', 'q validacija'], loc='lower center', markerscale=5,
                   bbox_to_anchor=(0.75, -0.35), fancybox=True)
        #plt.savefig('results/input_train_validation/' + input_plt_name)

        #col1, col2, col3 = st.columns([0.5, 3, 0.5])
        #with col2:
        #    st.pyplot(fig)
        #fig, ax = plt.subplots()
        #fig.set_size_inches(13, 8)
        axs[1].set_title('Gubitak treninga i validacije')
        axs[1].set_xlabel('Epoha')
        axs[1].set_ylabel('MSE gubitak')
        axs[1].plot(loss_list, color=(0.5, 0.4, 0.5, 0.6))
        axs[1].plot(loss_val_list, color=(0.1, 0.4, 0.5, 0.6))
        box = axs[1].get_position()
        axs[1].set_position([box.x0, box.y0 + box.height * 0.2,
                         box.width, box.height * 0.8])
        axs[1].legend(['Gubitak treninga', 'Gubitak validacije'], loc='lower center',
                    bbox_to_anchor=(0.25, -0.35), fancybox=True)
        #plt.savefig('results/losses/' + loss_plt_name)
        #col1, col2, col3 = st.columns([0.5, 3, 0.5])
        #with col2:
        #    st.pyplot(fig)

        #fig, ax = plt.subplots()
        #fig.set_size_inches(13, 8)
        axs[2].scatter([i for i in range(bsize*bdiv_all)], v2_comb.flatten(), s=0.2, color=(0.1, 0.2, 0.5))
        axs[2].plot(q_pred, color=(0.5, 0.2, 0.5))
        axs[2].set_title('Evaluacija modela strojnoga učenja')
        axs[2].set_xlabel('Vrijeme [minute]')
        axs[2].set_ylabel('Toplinski tok [$W / m^2$]')
        ax3 = axs[2].twinx()
        ax3.set_ylim([-7, 7])
        ax3.plot(u_m, color=(0.5, 0.4, 0.5, 0.4))
        ax3.plot(u_p, color=(0.1, 0.4, 0.5, 0.4))
        ax3.set_ylabel('U vrijednost [W/($m^2K$)]')
        box = axs[2].get_position()
        axs[2].set_position([box.x0, box.y0 + box.height * 0.2,
                         box.width, box.height * 0.8])
        axs[2].legend(['q predikcija', 'q mjereno'], loc='lower center', markerscale=5,
                  bbox_to_anchor=(0.25, -0.35), fancybox=True)
        ax3.legend(['U mjereno', 'U predikcija'], loc='lower center', markerscale=5,
                   bbox_to_anchor=(0.75, -0.35), fancybox=True)
        plt.savefig('results/plots/' + model_name + '.png', dpi=300)
        col1, col2, col3 = st.columns([0.5, 3, 0.5])
        with col2:
            st.pyplot(fig)

        print(U_m, U_p, abs_error)

        # Saving the model
        if model_save:
            model_name = 'results/models/' + model_name + '.pt'
            torch.save(model, model_name)

## pogledaj warning za torch tensor!
## Popravi da Run uvijek radi
