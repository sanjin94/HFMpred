import sys

from sklearn.preprocessing import quantile_transform
sys.path.append('data')
sys.path.append('icons')
sys.path.append('dl_models')

from PIL import Image
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import torch
import time
import regression as worker


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
    bsize = st.sidebar.number_input('Specify batch size', value=64)
    nlayers = st.sidebar.number_input('Specify number of LSTM layers', min_value=0, value=32)
    device = st.sidebar.selectbox('Specify execution device', ['cuda', 'cpu'])
    dropout = st.sidebar.number_input('Specify dropout probability', min_value=0.00, max_value=1.00, value=0.30)
    select_optimizer = st.sidebar.selectbox('Select optimizer', ['Adam', 'SGD'])
    nepochs = st.sidebar.number_input('Specify number of epochs', min_value=1, value=50)
    model_name = st.sidebar.text_input('Specify model name', value=file_name[0:-4]+'_LSTM_b'+str(bsize)+'_nl'+str(nlayers)+'_pd'+str(dropout)[2:]+'_op'+select_optimizer)
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
        if select_optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        elif select_optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
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

        input_plt_name = model_name + '_input'
        loss_plt_name = model_name + '_loss'
        results_plt_name = model_name + '_results'
        fig, ax = plt.subplots()
        fig.set_size_inches(13, 8)
        ax.set_title('Ulazni podaci')
        ax.set_xlabel('Vrijeme [minute]')
        ax.set_ylabel('Razlika temperature [°C]')
        ax2 = ax.twinx()
        ax2.set_ylabel('Toplinski tok [$W / m^2$]')
        ax.scatter(split_list[2], split_list[0], s=0.2, color=(0.5, 0.2, 0.5))
        ax2.scatter(split_list[2], split_list[1], s=0.2, color=(0.1, 0.2, 0.5))
        ax.scatter(split_list[5], split_list[3], s=0.2, color=(0.5, 0.4, 0.5, 0.4))
        ax2.scatter(split_list[5], split_list[4], s=0.2, color=(0.1, 0.4, 0.5, 0.4))
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.2,
                          box.width, box.height * 0.8])
        ax.legend(['$\Delta T$ trening', '$\Delta T$ validacija'], loc='lower center', markerscale=5,
                  bbox_to_anchor=(0.35, -0.35), fancybox=True)
        ax2.legend(['q trening', 'q validacija'], loc='lower center', markerscale=5,
                   bbox_to_anchor=(0.65, -0.35), fancybox=True)
        plt.savefig('results/input_train_validation/' + input_plt_name)

        col1, col2, col3 = st.columns([0.5, 3, 0.5])
        with col2:
            st.pyplot(fig)
        fig, ax = plt.subplots()
        fig.set_size_inches(13, 8)
        ax.set_title('Gubitak treninga i validacije')
        ax.set_xlabel('Epoha')
        ax.set_ylabel('MSE gubitak')
        ax.plot(loss_list, color=(0.5, 0.4, 0.5, 0.6))
        ax.plot(loss_val_list, color=(0.1, 0.4, 0.5, 0.6))
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.2,
                         box.width, box.height * 0.8])
        ax.legend(['Gubitak treninga', 'Gubitak validacije'], loc='lower center',
                    bbox_to_anchor=(0.5, -0.35), fancybox=True)
        plt.savefig('results/losses/' + loss_plt_name)
        col1, col2, col3 = st.columns([0.5, 3, 0.5])
        with col2:
            st.pyplot(fig)

        fig, ax = plt.subplots()
        fig.set_size_inches(13, 8)
        ax.scatter([i for i in range(bsize*bdiv_all)], v2_comb.flatten(), s=0.2, color=(0.1, 0.2, 0.5))
        ax.plot(q_pred, color=(0.5, 0.2, 0.5))
        u_m, U_m = worker.u_val(v1_comb.cpu().flatten().numpy(), v2_comb.cpu().flatten().numpy())
        u_p, U_p = worker.u_val(v1_comb.cpu().flatten().numpy(), q_pred)
        ax.set_xlabel('Vrijeme [minute]')
        ax.set_ylabel('Toplinski tok [$W / m^2$]')
        ax2 = ax.twinx()
        ax2.set_ylim([-7, 7])
        ax2.plot(u_m, color=(0.5, 0.4, 0.5, 0.4))
        ax2.plot(u_p, color=(0.1, 0.4, 0.5, 0.4))
        ax2.set_ylabel('U vrijednost [W/($m^2K$)]')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.2,
                         box.width, box.height * 0.8])
        ax.legend(['q predikcija', 'q mjereno'], loc='lower center', markerscale=5,
                  bbox_to_anchor=(0.35, -0.35), fancybox=True)
        ax2.legend(['U mjereno', 'U predikcija'], loc='lower center', markerscale=5,
                   bbox_to_anchor=(0.65, -0.35), fancybox=True)
        plt.savefig('results/predictions/' + results_plt_name)
        col1, col2, col3 = st.columns([0.5, 3, 0.5])
        with col2:
            st.pyplot(fig)
        print(abs((U_m - U_p) / U_p) * 100)

        # Saving the model
        if model_save:
            model_name = 'results/models/' + model_name + '.pt'
            torch.save(model, model_name)

## popravi prijelaz između vektora!
## pogledaj warning za torch tensor!
## Popravi da Run uvijek radi
