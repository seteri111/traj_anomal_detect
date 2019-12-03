import os.path
import numpy as np
import pandas as pd
from dateutil.parser import parse

def read_file_all(s):
    large_df = []
    file_root = 'datas/AIS_Data_txt'
    for root, dir, files in os.walk(file_root):
        for f in files:
            if os.path.splitext(f)[1] == '.csv':
                if 'Zone03' not in f:
                    continue
                k = 0
                for i in range(s):
                    if str(i + 1) not in f:
                        k = 1
                if k:
                    continue
                try:
                    file_path = os.path.join(root, f)
                    large_df.append(
                            pd.read_csv(file_path,
                                        usecols=[0, 1, 2, 3, 4, 5],
                                        engine='python'))
                    print('%s load!' % os.path.splitext(f)[0])
                except:
                    print('error at: ' + f)
    df = pd.concat(large_df)
    df.rename(columns={'MMSI':'id'},inplace=True)
    return df.sort_values('id')

def remove_stationary_points(df):
    df = df[df.SOG != 0]
    return df

def time_resample(df):
    df['BaseDateTime'] = df['BaseDateTime'].apply(lambda x:parse(x))
    df = df.sort_values('BaseDateTime')
    df = df.drop_duplicates(subset='BaseDateTime', keep='first')
    df = df.reset_index()
    n = 0
    for i in range(df.shape[0] - 1):
        timedelta = df.loc[i + 1, 'BaseDateTime'] - df.loc[n, 'BaseDateTime']
        if timedelta.total_seconds() < 1800:
            df.loc[i + 1, 'SOG'] = 0
            continue
        n = i + 1

    df = remove_stationary_points(df)
    return df[['id', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG']]

def pre_process(df):
    df = remove_stationary_points(df)
    df = df.dropna(how='any', axis=0)
    df_a = pd.DataFrame()

    for idx in df['id'].drop_duplicates():
        t = df[df.id == idx]
        if t.shape == (6, ):
            continue
        df_a = pd.concat([df_a, time_resample(t)], axis=0)
        del t
        print('Resample of %s has done!' % idx)
    del df

    print('Resample end.')
    return df_a

def write_tofile(df, s):
    fileroot = 'datas'
    f = fileroot + os.sep + 'AIS_cvs_file_' + s + '.txt'
    df.to_csv(f, mode='w', index=False)
    print(f + ' was written')

if __name__ == "__main__":
    s = input('Input months you want:')
    df = read_file_all(int(s))
    write_tofile(pre_process(df), s)
