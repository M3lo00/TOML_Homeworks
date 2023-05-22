from sklearn.svm import SVR
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import itertools
from tqdm import tqdm
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')

# Load the CSV file
data = pd.read_csv("BC-Data-Set.csv")

# Convert the date column to a datetime object
data['date'] = pd.to_datetime(data['date'])

# Remove any missing values
data = data.dropna()

# Set the date column as the index of the DataFrame
data = data.set_index('date')

C, gamma = (46.41588833612773, 0.021544346900318832)

seed = 42

threshold = 6 # theshold a little high to retain some outliers
z_scores = np.abs((data - data.mean()) / data.std())
outliers = (z_scores > threshold).any(axis=1)
for column in data.columns:
    column_median = data[column].median()
    data.loc[outliers, column] = column_median

x_train, x_val, y_train, y_val = train_test_split(data, data.BC, test_size=0.3, random_state=seed, shuffle=True)
x_train = x_train.drop(columns=['BC'])
x_val = x_val.drop(columns=['BC'])


import sys
K = sys.argv[1]
K = int(K)
all_cols = [c for c in list(x_train.columns.values)]
assert 1 <= K and K <= len(all_cols)
print(f'>>> {K=}')

cols = [list(x) for x in itertools.combinations(all_cols, K)]

scaler = StandardScaler()
res = []
for c in tqdm(cols):
    _x_train = x_train.drop(columns=c) 
    _x_val = x_val.drop(columns=c)

    clf = SVR(C=C, gamma=gamma, kernel='rbf')
    clf.fit(scaler.fit_transform(_x_train), y_train)
    r2 = clf.score(scaler.transform(_x_val), y_val)
    y_hat = clf.predict(scaler.transform(_x_val))
    rmse = metrics.mean_squared_error(y_val, y_hat, squared=False)
    res.append({
        'dropping': c,
        'remaining_cols': list(_x_train.columns.values),
        'r2': r2,
        'rmse': rmse,
    })


print(res)
pd.DataFrame(res).to_csv(f'_{K}_select.csv')
print('>>> Saved')
