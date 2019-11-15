import numpy as np
import pandas as pd
import os

def get_AIS_csv_data(f):
    if not os.path.exists(f):
        from import_csv import get_csv_data
        get_csv_data()
        
    df = pd.read_csv(f,
                        usecols=[0, 1, 2, 3],
                        parse_dates=['BaseDateTime'],
                        engine='python')
    return df

def partition_AIS(df, sz):
    dfs = []
    sz = int(sz)
    lon_max = df['LON'].max()
    lon_min = df['LON'].min()
    lon_stp = (lon_max - lon_min)/sz
    lat_max = df['LAT'].max()
    lat_min = df['LAT'].min()
    lat_stp = (lat_max - lat_min)/sz

    del lon_max, lat_max
    
    for i in range(sz):
        lon = lon_min + lon_stp * i
        for j in range(sz):
            lat = lat_min + lat_stp * j
            dft = df[
                (df.LON >= lon) & (df.LON < lon + lon_stp) &
                (df.LAT >= lat) & (df.LAT < lat + lat_stp)
            ]
            if dft.empty:
                continue
            dfs.append(dft)
            print('Partition %d has done' % (sz * i + j + 1))

    return dfs

def write_to_file(s, dfs):
    i = 1
    root = './AIS_cell'
    for df in dfs:
        f = 'AIS_' + s + '_part_' + str(i) + '.txt'
        file_path = root + os.sep + f
        df.to_csv(file_path, mode='w', index=False)
        i += 1

s = input('Input month length:')
f = 'AIS_cvs_file_' + s + '.txt'
sz = input('Input partition size:')
df = get_AIS_csv_data(f)
del f
write_to_file(s, partition_AIS(df, sz))
