import glob
import pandas as pd

dir_pattern = '/home/biadmin/data/cuvalley/zad3/*'
files = sorted(glob.glob(dir_pattern))

# column - id mapping
tmp_df = pd.read_csv(files[0])
mapping = list(zip(range(len(tmp_df.columns)), tmp_df.columns))
mapping_dct = {m[1]: m[0] for m in mapping}

dfs = []
for file in files:
    dfs.append(pd.read_csv(file))

rows_cnt = sum([len(df) for df in dfs])

df = pd.concat(dfs, ignore_index=True)

assert len(df) == rows_cnt

df.shape

df = df.rename(columns=mapping_dct)

df.to_csv('/home/biadmin/data/cuvalley/data_X_raw.csv', index=False)

####

y_input_path = '/home/biadmin/data/cuvalley/temp_zuz.csv'

df_y = pd.read_csv(y_input_path, sep=';')

df_y.shape

df_y = df_y.rename(columns={'Czas': 0, 'temp_zuz': 58})

df_y.to_csv('/home/biadmin/data/cuvalley/data_y_raw.csv', index=False)
