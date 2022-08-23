import pandas as pd

df = pd.read_csv('data/raw/TRSYS01_Public.dat',
                 skiprows=1, usecols=['TIMESTAMP'])

start = '2021-10-17 12:48'
end = '2021-10-17 13:22'
lt = int(len(start))

for ind in df.index:
    if str(df['TIMESTAMP'][ind])[0:lt] == start:
        print(ind)
    if str(df['TIMESTAMP'][ind])[0:lt] == end:
        print(ind)
