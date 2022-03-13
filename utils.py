import pandas as pd


def read_ready_to_shift(col: str) -> pd.DataFrame:
    tmp = pd.read_csv('/home/biadmin/data/cuvalley/ready_to_shift.csv')
    tmp['0'] = pd.to_datetime(tmp['0'])
    return tmp[['0', col]]


def read_shifted(col: str) -> pd.DataFrame:
    tmp = pd.read_csv(f'/home/biadmin/data/cuvalley/shifted/{col}.csv')
    tmp.drop(columns=['Unnamed: 0'], inplace=True)
    tmp['0'] = pd.to_datetime(tmp['0'])
    return tmp


def read_x_raw() -> pd.DataFrame:
    x_path = '/home/biadmin/data/cuvalley/data_X_raw.csv'
    x_df = pd.read_csv(x_path)
    x_df['0'] = x_df['0'].str.split('+').str[0]
    x_df['0'] = pd.to_datetime(x_df['0'])

    return x_df


def read_y_raw() -> pd.DataFrame:
    y_path = '/home/biadmin/data/cuvalley/data_y_raw.csv'
    y_df = pd.read_csv(y_path)
    y_df['0'] = pd.to_datetime(y_df['0'])
    return y_df


def read_merged_df() -> pd.DataFrame:
    tmp = pd.read_csv('/home/biadmin/data/cuvalley/data_merged_outer_raw.csv')
    tmp['0'] = pd.to_datetime(tmp['0'])
    return tmp
