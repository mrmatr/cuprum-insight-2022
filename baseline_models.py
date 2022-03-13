import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

x_path = '/home/biadmin/data/cuvalley/data_X_raw.csv'
y_path = '/home/biadmin/data/cuvalley/data_y_raw.csv'

x_df = pd.read_csv(x_path)
y_df = pd.read_csv(y_path)

# parse datetime str
print(type(x_df.loc[0, '0']))  # <class 'str'>
print(type(y_df.loc[0, '0']))  # <class 'str'>

x_df['0'] = x_df['0'].str.split('+').str[0]

x_df['0'] = pd.to_datetime(x_df['0'])
y_df['0'] = pd.to_datetime(y_df['0'])

x_df[['0', '1']].info()
y_df[['0', '58']].info()

print(x_df.shape)  # (702780, 58)
print(y_df.shape)  # (11384, 2)

# merge
merged_df = pd.merge(y_df, x_df, on='0', how='left')

print(merged_df.shape)  # (11389, 59)

merged_df: pd.DataFrame = merged_df.groupby('0').mean().reset_index()

merged_df[merged_df.isna().any(axis=1)]
#                         0    58   1   2   3   4   5  ...  51  52  53  54  55  56  57
# 9461  2021-11-09 17:10:10  1308 NaN NaN NaN NaN NaN  ... NaN NaN NaN NaN NaN NaN NaN
# 11383 2022-02-01 00:00:00  1305 NaN NaN NaN NaN NaN  ... NaN NaN NaN NaN NaN NaN NaN
# [2 rows x 59 columns]


merged_df = merged_df.dropna()

print(merged_df.shape)  # (11382, 59)

# corr
merged_df.corr()

# 1. BASELINE

cols_X = list(set(merged_df.columns).difference(['0', '1', '2', '3', '4', '58']))
cols_y = ['58']

chunk_results = []
kf = KFold(n_splits=5, random_state=1, shuffle=True)
for train_indexes, test_indexes in kf.split(merged_df):
    train, test = merged_df.iloc[train_indexes], merged_df.iloc[test_indexes]
    train_X, train_y = train[cols_X], train[cols_y]
    test_X, test_y = test[cols_X], test[cols_y]

    lr = LinearRegression()

    lr.fit(train_X, train_y)

    preds = lr.predict(test_X)

    mse = mean_squared_error(test_y, preds)
    r2 = r2_score(test_y, preds)
    mae = mean_absolute_error(test_y, preds)

    chunk_results.append({'mse': mse, 'r2': r2, 'mae': mae})

pd.DataFrame(chunk_results)
#          mse        r2       mae
# 0  42.919461  0.193609  4.744306
# 1  36.932305  0.236162  4.415017
# 2  54.593924  0.242637  4.826164
# 3  42.974534  0.213322  4.629018
# 4  44.207526  0.200081  4.569677


merged_df['58'].describe()
# merged_df['58'].describe()
# count    11382.000000
# mean      1303.388772
# std          7.532425
# min       1190.000000
# 25%       1300.000000
# 50%       1304.000000
# 75%       1308.000000
# max       1338.000000
# Name: 58, dtype: float64

# 2. T - 1

y_df_ = y_df.copy()  # copy original df

y_df['0_shift_1'] = y_df['0'] + pd.DateOffset(hours=1)
y_df = y_df.rename(columns={'58': '58_t-1'})
y_df = y_df.drop(columns=['0'])

merged_df = pd.merge(merged_df, y_df, left_on='0', right_on='0_shift_1', how='inner')

cols_X = sorted(set(merged_df.columns).difference(['0', '1', '2', '3', '4', '58', '0_shift_1']))
cols_y = ['58']

chunk_results = []
kf = KFold(n_splits=5, random_state=1, shuffle=True)
for train_indexes, test_indexes in kf.split(merged_df):
    train, test = merged_df.iloc[train_indexes], merged_df.iloc[test_indexes]
    train_X, train_y = train[cols_X], train[cols_y]
    test_X, test_y = test[cols_X], test[cols_y]

    lr = LinearRegression()

    lr.fit(train_X, train_y)

    preds = lr.predict(test_X)

    mse = mean_squared_error(test_y, preds)
    r2 = r2_score(test_y, preds)
    mae = mean_absolute_error(test_y, preds)

    chunk_results.append({'mse': mse, 'r2': r2, 'mae': mae})

pd.DataFrame(chunk_results)
#         mse        r2       mae
# 0  27.801703  0.439562  3.752727
# 1  26.494473  0.447704  3.823871
# 2  29.346373  0.412792  3.862016
# 3  34.565508  0.394715  3.953920
# 4  27.438863  0.419850  3.771344

merged_df['58'].describe()

# 3. nn

chunk_results = []
kf = KFold(n_splits=5, random_state=1, shuffle=True)
for train_indexes, test_indexes in kf.split(merged_df):
    train, test = merged_df.iloc[train_indexes], merged_df.iloc[test_indexes]
    train_X, train_y = train[cols_X], train[cols_y]['58']
    test_X, test_y = test[cols_X], test[cols_y]['58']

    # nn = MLPRegressor(hidden_layer_sizes=(30), max_iter=1000, validation_fraction=0.2, early_stopping=True)
    nn = MLPRegressor(hidden_layer_sizes=(30), max_iter=1000, validation_fraction=0.2)

    nn.fit(train_X, train_y)

    preds = nn.predict(test_X)

    mse = mean_squared_error(test_y, preds)
    r2 = r2_score(test_y, preds)
    mae = mean_absolute_error(test_y, preds)

    chunk_results.append({'mse': mse, 'r2': r2, 'mae': mae})

pd.DataFrame(chunk_results)
#           mse        r2        mae
# 0   90.749103 -0.829357   7.709027
# 1   50.404306 -0.050713   5.210565
# 2  270.176937 -4.406123  15.343956
# 3   51.205534  0.103328   4.912082
# 4   42.572553  0.099873   4.847100

# without cv

# merged_df.to_csv('/home/biadmin/data/cuvalley/merged_df_v01.csv')

X = merged_df[cols_X]
y = merged_df[cols_y[0]]

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=1)
print(train_X.shape, test_X.shape, train_y.shape, test_y.shape)  # (8810, 54) (2203, 54) (8810,) (2203,)

train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.1, random_state=1)
print(train_X.shape, valid_X.shape, train_y.shape, valid_y.shape)  # (7929, 54) (881, 54) (7929,) (881,)

print(train_y.shape, test_y.shape, valid_y.shape)  # (7929,) (2203,) (881,)

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

train_X = scaler_x.fit_transform(train_X)
train_y = scaler_y.fit_transform(train_y.values.reshape(-1, 1))[:, 0]

valid_X = scaler_x.transform(valid_X)
test_X = scaler_x.transform(test_X)

mlp = MLPRegressor(hidden_layer_sizes=(30), max_iter=1000)

train_scores = []
valid_scores = []
for epoch in range(1000):
    mlp.partial_fit(train_X, train_y)

    train_scores.append(mean_squared_error(scaler_y.inverse_transform(train_y.reshape(-1, 1))[:, 0],
                                           scaler_y.inverse_transform(mlp.predict(train_X).reshape(-1, 1))[:, 0]))
    valid_scores.append(
        mean_squared_error(valid_y, scaler_y.inverse_transform(mlp.predict(valid_X).reshape(-1, 1))[:, 0]))

test_pred = scaler_y.inverse_transform(mlp.predict(test_X).reshape(-1, 1))[:, 0]
print(mean_squared_error(test_y, test_pred), r2_score(test_y, test_pred), mean_absolute_error(test_y, test_pred))
# 25.051543239855196 0.4950008599016652 3.6272222367992604

plt.plot(range(len(train_scores)), train_scores, label='train')
plt.plot(range(len(valid_scores)), valid_scores, label='valid')
plt.yscale('log')
plt.legend()

#

X = merged_df[cols_X]
y = merged_df[cols_y[0]]

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=1)
print(train_X.shape, test_X.shape, train_y.shape, test_y.shape)  # (8810, 54) (2203, 54) (8810,) (2203,)

train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.1, random_state=1)
print(train_X.shape, valid_X.shape, train_y.shape, valid_y.shape)  # (7929, 54) (881, 54) (7929,) (881,)

print(train_y.shape, test_y.shape, valid_y.shape)  # (7929,) (2203,) (881,)

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

train_X = scaler_x.fit_transform(train_X)
train_y = scaler_y.fit_transform(train_y.values.reshape(-1, 1))[:, 0]

valid_X = scaler_x.transform(valid_X)
test_X = scaler_x.transform(test_X)

mlp = MLPRegressor(hidden_layer_sizes=(30), max_iter=1000 * 100000)

train_scores = []
valid_scores = []

mlp.fit(train_X, train_y)

test_pred = scaler_y.inverse_transform(mlp.predict(test_X).reshape(-1, 1))[:, 0]
print(mean_squared_error(test_y, test_pred), r2_score(test_y, test_pred), mean_absolute_error(test_y, test_pred))
# 43.320788357783655 0.12672202827576562 4.809432349469878

plt.plot(range(len(train_scores)), train_scores, label='train')
plt.plot(range(len(valid_scores)), valid_scores, label='valid')
plt.yscale('log')
plt.legend()

# Mlp cv scale


chunk_results = []
kf = KFold(n_splits=5, random_state=1, shuffle=True)
for train_indexes, test_indexes in kf.split(merged_df):
    train, test = merged_df.iloc[train_indexes], merged_df.iloc[test_indexes]
    train_X, train_y = train[cols_X], train[cols_y]['58']
    test_X, test_y = test[cols_X], test[cols_y]['58']

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    train_X_scaled = scaler_x.fit_transform(train_X)
    train_y_scaled = scaler_y.fit_transform(train_y.values.reshape(-1, 1))[:, 0]

    test_X_scaled = scaler_x.transform(test_X)

    # nn = MLPRegressor(hidden_layer_sizes=(30), max_iter=1000, validation_fraction=0.2, early_stopping=True)
    nn = MLPRegressor(hidden_layer_sizes=(50), max_iter=1000)

    nn.fit(train_X_scaled, train_y_scaled)

    preds = scaler_y.inverse_transform(nn.predict(test_X_scaled).reshape(-1, 1))[:, 0]

    mse = mean_squared_error(test_y, preds)
    r2 = r2_score(test_y, preds)
    mae = mean_absolute_error(test_y, preds)

    chunk_results.append({'mse': mse, 'r2': r2, 'mae': mae})

pd.DataFrame(chunk_results)
#          mse        r2       mae
# 0  36.484576  0.264529  4.458108
# 1  42.039424  0.123659  4.873787
# 2  38.758634  0.224457  4.582151
# 3  45.866613  0.196819  4.556724
# 4  37.388257  0.209486  4.767329
