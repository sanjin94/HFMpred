import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np

# room d137b #

file = open("data_raw/_d137b.csv")
read = csv.reader(file, delimiter=';')
_read = pd.DataFrame(list(read)).to_numpy()
data_start = 1
data_end = 511
e2_q = _read[:, 1][data_start:data_end].astype(np.float) * -1
e2_ti = _read[:, 2][data_start:data_end].astype(np.float)
e2_te = _read[:, 3][data_start:data_end].astype(np.float)
file.close()
