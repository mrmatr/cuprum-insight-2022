from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model import create_rnn
from utils import read_merged_df

df = read_merged_df()

# FEATURE SELECTION
# sumujemy nadawy
df['1_2_nad'] = df['1'] + df['2']
# df.drop(columns=['1', '2'], inplace=True)

# mnożenie
df['1_2_nad_55_corg'] = df['1_2_nad'] * df['55']
df['1_2_nad_56_fe'] = df['1_2_nad'] * df['56']
df['1_2_nad_57_s'] = df['1_2_nad'] * df['57']

# kolektory
# średnia z temperatur in
df['51_52_water_in_temp'] = df[['51', '52']].mean(axis=1)

# róznica temperatur out - in
df['14_water_temp_diff'] = df['14'] - df['51_52_water_in_temp']
df['15_water_temp_diff'] = df['15'] - df['51_52_water_in_temp']
df['16_water_temp_diff'] = df['16'] - df['51_52_water_in_temp']
df['17_water_temp_diff'] = df['17'] - df['51_52_water_in_temp']
df['18_water_temp_diff'] = df['18'] - df['51_52_water_in_temp']
df['19_water_temp_diff'] = df['19'] - df['51_52_water_in_temp']
df['20_water_temp_diff'] = df['20'] - df['51_52_water_in_temp']
df['21_water_temp_diff'] = df['21'] - df['51_52_water_in_temp']

# przemnazamy przez predkość przeplywu
df['14_energy_loss'] = df['14_water_temp_diff'] * df['5']
df['15_energy_loss'] = df['15_water_temp_diff'] * df['6']
df['16_energy_loss'] = df['16_water_temp_diff'] * df['7']
df['17_energy_loss'] = df['17_water_temp_diff'] * df['8']
df['18_energy_loss'] = df['18_water_temp_diff'] * df['9']
df['19_energy_loss'] = df['19_water_temp_diff'] * df['10']
df['20_energy_loss'] = df['20_water_temp_diff'] * df['11']
df['21_energy_loss'] = df['21_water_temp_diff'] * df['12']

df['energy_loss'] = df[[c for c in df.columns if '_energy_loss' in c]].sum(axis=1)

# wybor zmiennych do modelu
cols = [
    '0',  # czas
    '1_2_nad',  # nadawy
    '3',
    '4',
    '13',
    'energy_loss',
    '1_2_nad_55_corg',
    '1_2_nad_56_fe',
    '1_2_nad_57_s',
    '58'  # target - temp zuzla
]

# Prepare dataset with datetime index
data = df[cols]

ts_series = pd.to_datetime(data['0'])
datetime_index = pd.DatetimeIndex(ts_series.values)

ts = data.set_index(datetime_index)
ts = ts.drop('0', axis=1)

ts.sort_index(inplace=True)

ts = ts.resample('60s').mean()

ts_minute = ts.asfreq('60s')

# Add t-1 (using fill backward 'hack')
ts_minute['58_t-1'] = ts_minute[~ts_minute['58'].isna()]['58'].asfreq('60s',  method='ffill').shift(1)

# remove unstable rows
unstable_rows = pd.to_datetime(pd.read_csv('unstable_rows.csv', header=None)[0])

ts_minute['date_'] = ts_minute.index.date.astype(str)
ts_minute = ts_minute[~ts_minute['date_'].isin(unstable_rows.astype(str))]

# SCALE VARIABLES
x_cols = cols = ['1_2_nad', '3', '4', '13', 'energy_loss', '1_2_nad_55_corg', '1_2_nad_56_fe', '1_2_nad_57_s', '58_t-1']
y_col = '58'

scaler_x = StandardScaler()
scaler_y = StandardScaler()

raw_x = ts_minute[x_cols]
raw_y = ts_minute[y_col]
scaled_x = scaler_x.fit_transform(raw_x)
scaled_y = scaler_y.fit_transform(raw_y.values.reshape(-1, 1))

scaled_tf = np.concatenate((scaled_x, scaled_y), axis=1)

frame_indexes = [(ub[0] - 60 + 1, ub[0] + 1) for ub in np.argwhere(~np.isnan(scaled_tf[:, -1]))[1:]]

frames = [scaled_tf[f[0]: f[1]] for f in frame_indexes]


def is_correct_frame(frame: np.ndarray) -> bool:
    return np.isnan(frame[:, :-1]).sum() == 0


frames_correct = [frame for frame in frames if is_correct_frame(frame)]
len([frame for frame in frames if not is_correct_frame(frame)])  # incorrect frames num


def create_x_y_frames(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = frame[:, :-1]
    y = np.array(frame[-1, -1])
    return x, y


splitted_frames = [create_x_y_frames(frame) for frame in frames_correct]

X = [f[0] for f in splitted_frames]
y = [f[1] for f in splitted_frames]

X = np.array(X)
y = np.array(y)

X.shape
y.shape

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=1)
train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.1, random_state=1)

print(train_X.shape, valid_X.shape, test_X.shape)  # (8192, 60, 8) (911, 60, 8) (2276, 60, 8)

input_shape = train_X[0].shape

rnn = create_rnn(5, 1, input_shape, ['relu', 'linear'])


epochs = 100
history = rnn.fit(train_X, train_y, batch_size=100, epochs=epochs, verbose=2, validation_data=(valid_X, valid_y))


plt.plot(np.arange(epochs), history.history['mean_squared_error'], label='train_mse')
plt.plot(np.arange(epochs), history.history['val_mean_squared_error'], label='valid_mse')
plt.legend()


preds_scaled = rnn.predict(test_X)
preds_raw = scaler_y.inverse_transform(preds_scaled)

mean_squared_error(scaler_y.inverse_transform(test_y.reshape(-1,1)), preds_raw)

mean_squared_error(scaler_y.inverse_transform(test_y.reshape(-1,1)), preds_raw)**(1/2)