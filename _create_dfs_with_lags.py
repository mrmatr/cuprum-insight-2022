import pandas as pd
from tqdm import tqdm

from utils import read_ready_to_shift

# cols_to_lag = ['3', '13', '55', '56', '57', '1_2_nad', '1_2_nad_55_corg', '1_2_nad_56_fe', '1_2_nad_57_s']
cols_to_lag = ['55']
lag_numbers = 60


def prepare_lagged_col(original_df: pd.DataFrame, col: str, lag: int) -> pd.DataFrame:
    time_shift_col = f'0_shift_{lag}'
    shifted_val_col = f'{col}_t-{lag}'

    to_merge = original_df[['0', col]]
    to_merge[time_shift_col] = to_merge['0'] + pd.DateOffset(minutes=lag)
    to_merge = to_merge.rename(columns={ctl: shifted_val_col}).drop(columns='0')

    return to_merge, time_shift_col, shifted_val_col


def add_lagged_cols(original_df: pd.DataFrame, to_append_df: pd.DataFrame, col: str, lags: int):
    to_append_df['0_int'] = to_append_df['0'].astype(int)
    to_append_df.set_index('0_int', inplace=True)
    for lag in tqdm(range(1, lags + 1)):
        to_merge, time_shift_col, _ = prepare_lagged_col(original_df, col, lag)
        to_merge[f'{time_shift_col}'] = to_merge[time_shift_col]  # todo
        to_merge.set_index(time_shift_col, inplace=True)
        to_append_df = to_append_df.join(to_merge, how='inner')

    return to_append_df


for ctl in tqdm(cols_to_lag):
    df = read_ready_to_shift(ctl)

    # df_with_shifted_cols = add_lagged_cols(df, df.copy(), ctl, lag_numbers)
    df_with_shifted_cols = add_lagged_cols(df, df.copy(), ctl, lag_numbers)

    df_with_shifted_cols.to_csv(f'/home/biadmin/data/cuvalley/shifted/{ctl}.csv')

print('Done')
