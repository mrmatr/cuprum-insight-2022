from utils import read_x_raw, read_y_raw
import pandas as pd

x_df = read_x_raw()
y_df = read_y_raw()

merged_df = pd.merge(x_df, y_df, on='0', how='outer')

merged_df.to_csv('/home/biadmin/data/cuvalley/data_merged_outer_raw.csv', index=False)

tmp = pd.read_csv('/home/biadmin/data/cuvalley/data_merged_outer_raw.csv')

tmp[['0', '1']].info()
