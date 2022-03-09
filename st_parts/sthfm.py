import sys
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


#EMOJI_ICON = "icons/HFMpred.ico"
#EMOJI_PNG = "icons/HFMpred.png"

#st.set_page_config(page_title="HFMpred", page_icon=EMOJI_ICON,
#                   layout='wide')

#col1, col2 = st.columns(2)
#with col2:
#    st.image(EMOJI_PNG, width=80)
#col1, col2 = st.columns([1.5,6])
#with col2:
#    st.title('HFMpred - a tool for HFM results analysis and prediction')
#col1, col2 = st.columns([5, 6])
#with col2:
#    st.markdown('## U-value analysis')

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
    dropout = st.sidebar.number_input('Specify dropout probability', min_value=0.00, max_value=1.00, value=0.20)
    select_optimizer = st.sidebar.selectbox('Select optimizer', ['Adam', 'SGD'])
    nepochs = st.sidebar.number_input('Specify number of epochs', min_value=1, value=50)
    model_name = st.sidebar.text_input('Specify model name', value=file_name[0:-4]+'_LSTM_b'+str(bsize)+'_nl'+str(nlayers)+'_pd'+str(dropout)[2:]+'_op'+select_optimizer)
    avg_const = st.sidebar.number_input('Specify average constant:', min_value=1, max_value=60, value=10)
    regularization = st.sidebar.checkbox('Regularization', value=True)
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

        u_list = []
        for i in range(len(v1_train.flatten())):
            n = i + 1
            deltat = v1_train.flatten().cpu().numpy()
            q = v2_train.flatten().cpu().numpy()
            deltat = np.sum(deltat[0:n]) / (n + 1)
            q = np.sum(q[0:n]) / (n + 1)
            u = q / deltat
            u_list.append(u)

        u_train = torch.tensor(u_list).view(bdiv, 1, bsize).to(device).float()

        u_list = []
        for i in range(len(v1_validation)):
            n = i + 1
            deltat = v1_validation.copy()
            q = v2_validation.copy()
            deltat = np.sum(deltat[0:n]) / (n + 1)
            q = np.sum(q[0:n]) / (n + 1)
            u = q / deltat
            u_list.append(u)

        u_validation = np.array(u_list)

        #regulizer = u_train

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
                if regularization == True:
                    l2_lambda = 0.001
                    l2_norm = torch.tensor(sum(p.pow(2.0).sum() for p in model.parameters())).to(device).float()
                    loss += l2_lambda * l2_norm
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

        fig, ax = plt.subplots()
        fig.set_size_inches(8, 5)
        ax.scatter(split_list[2], split_list[0], s=0.2, color=(0.5, 0.2, 0.5))
        ax.scatter(split_list[2], split_list[1], s=0.2, color=(0.1, 0.2, 0.5))
        ax.scatter(split_list[5], split_list[3], s=0.2, color=(0.5, 0.2, 0.5, 0.2))
        ax.scatter(split_list[5], split_list[4], s=0.2, color=(0.1, 0.2, 0.5, 0.2))

        col1, col2 = st.columns([3, 3])
        with col1:
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            fig.set_size_inches(8, 5)
            ax.plot(loss_list, color=(0.5, 0.2, 0.5))
            ax.plot(loss_val_list, color=(0.1, 0.2, 0.5, 0.6))
            st.pyplot(fig)

        fig, ax = plt.subplots()
        fig.set_size_inches(13, 8)
        ax.scatter(x1, v2_train[0:len(x1)], s=0.2, color=(0.1, 0.2, 0.5))
        ax.plot(x1, prediction_train[0:len(x1)])
        ax.scatter(x2, v2_validation[0:len(x2)], s=0.2, color=(0.1, 0.2, 0.5, 0.4))
        ax.plot(x2, prediction_validation[0:len(x2)])

        st.pyplot(fig)

        # Saving the model
        model_name = 'results/models/' + model_name + '.pt'
        torch.save(model, model_name)

## popravi prijelaz između vektora!
## pogledaj warning za torch tensor!
## Popravi da Run uvijek radi
