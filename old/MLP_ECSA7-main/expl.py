import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

import data

# Generating the data
train_validation = 2/3

ti_train = data.e2_ti[0:int(np.round(train_validation * len(data.e2_ti)))]
te_train = data.e2_te[0:int(np.round(train_validation * len(data.e2_ti)))]
q_train = data.e2_q[0:int(np.round(train_validation * len(data.e2_ti)))]

ti_validation = data.e2_ti[int(np.round(train_validation * len(data.e2_ti))):len(data.e2_ti)]
te_validation = data.e2_te[int(np.round(train_validation * len(data.e2_ti))):len(data.e2_ti)]
q_validation = data.e2_q[int(np.round(train_validation * len(data.e2_ti))):len(data.e2_ti)]

merged_array = np.stack([ti_train, te_train], axis=1)
merged_array1 = np.stack([ti_validation, te_validation], axis=1)

filepath = './saved_model_e2_2_3_1l'
model = load_model(filepath, compile=True)

predictions = model.predict(merged_array)
predictions1 = model.predict(merged_array1)

# plt.plot(data.e2_te)
# plt.plot(data.e2_ti)
# plt.plot(data.e2_q)
result = np.append(predictions, predictions1)
# plt.plot(result)
# plt.show()

ti_mean = np.mean(data.e2_ti)
te_mean = np.mean(data.e2_te)
q_mean = np.mean(data.e2_q)

U_res = abs(q_mean) / (abs(ti_mean - te_mean))

result_mean = np.mean(result)

U_mlp = abs(result_mean) / (abs(ti_mean - te_mean))

print("U from results:", U_res, "\nU from MLP: ", U_mlp)
print("Relative difference between those is: ",
      abs(np.round((100 * (U_res - U_mlp) / U_res), 2)), "%")
print("Nubmer of elements: ", len(data.e2_ti))
print("Train elements: ", len(data.e2_ti[0:int(np.round(train_validation * len(data.e2_ti)))]))
print("Validation elements: ", len(
    data.e2_ti[int(np.round(train_validation * len(data.e2_ti))):len(data.e2_ti)]))
