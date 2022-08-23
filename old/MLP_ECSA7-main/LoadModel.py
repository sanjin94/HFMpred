import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

import data

# Generating the data
train_validation = 1/4

ti_train = data.e2_ti[0:int(np.round(train_validation * len(data.e2_ti)))]
te_train = data.e2_te[0:int(np.round(train_validation * len(data.e2_ti)))]
q_train = data.e2_q[0:int(np.round(train_validation * len(data.e2_ti)))]

ti_validation = data.e2_ti[int(np.round(train_validation * len(data.e2_ti))):len(data.e2_ti)]
te_validation = data.e2_te[int(np.round(train_validation * len(data.e2_ti))):len(data.e2_ti)]
q_validation = data.e2_q[int(np.round(train_validation * len(data.e2_ti))):len(data.e2_ti)]

merged_array = np.stack([ti_train, te_train], axis=1)
merged_array1 = np.stack([ti_validation, te_validation], axis=1)

filepath = './saved_model_e2_1_4_1l3'  # ANN model 1_4, 1_2 or 2_3
model = load_model(filepath, compile=True)

predictions = model.predict(merged_array)
predictions1 = model.predict(merged_array1)

plt.rcParams["figure.figsize"] = (13, 8)
# plt.rcParams["legend.loc"] = 'upper left'
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.plot(data.e2_te, label="Exterior temperature $[°C]$")
plt.plot(data.e2_ti, label="Interior temperature $[°C]$")
plt.plot(data.e2_q, label="HFM heat flux $[W/m^2]$")
result = np.append(predictions, predictions1)
plt.plot(result, label="MLP heat flux $[W/m^2]$")
plt.ylabel("Heat flux $[W/m^2]$ / Temperature $[°C]$", fontsize=14)
plt.xlabel("Measuring samples (sampling = every 10 minutes)", fontsize=14)
plt.legend(bbox_to_anchor=(0.59, 0.85), fontsize=14, ncol=2)
plt.axvline(x=len(data.e2_ti[0:int(np.round(train_validation * len(data.e2_ti)))]), color='m')
plt.savefig('mlp_e2_1_4_1l3.png', format='png', dpi=600)  # ANN model 1_4, 1_2 or 2_3
plt.show()
