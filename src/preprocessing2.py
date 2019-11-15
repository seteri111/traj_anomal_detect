import numpy as np
import pandas as pd
import os.path
from geopy.distance import great_circle


def read_file_all():
    file_a = []
    file_root = 'datas/release/taxi_log_2008_by_id'
    for root, dir, files in os.walk(file_root):
        for file in files:
            file_a.append(os.path.join(root, file))
    return file_a


def read_file_sub():
    file_a = []
    file_root = 'datas/release/taxi_log_2008_by_id'
    for root, dir, files in os.walk(file_root):
        for file in files:
            # testing in small number of data
            if int(file.split('.')[0]) > 99:
                continue
            file_a.append(os.path.join(root, file))
    return file_a


def write_tofile(df, s):
    fileroot = 'datas'
    f = fileroot + os.sep + s + '.txt'
    df.to_csv(f, mode='a', index=False)



def add_feat(df_input):
    """
    This function takes in raw LAT LON BaseDateTime series from the microsoft geolife data.
    Preprocessing notes: skip the first six lines before doing pandas read csv , 
    expecting columns in ['id', 'LAT', 'LON', 'BaseDateTime']
    Requres: pandas imported as pd
                from vincenty import vincenty
         
    Adds:
        speed
    """
    df = df_input
    # add some initial shifts
    df['time_shift'] = df.BaseDateTime.shift(-1)
    df['long_shift'] = df.LON.shift(-1)
    df['lat_shift'] = df.LAT.shift(-1)

    # add speed
    def speed(x):
        try:
            s = great_circle(
                (x.lat_shift, x.long_shift),
                (x.LAT, x.LON)).meters / (x.time_shift -
                                           x.BaseDateTime).total_seconds()
            s *= 3.6
        except:
            s = np.nan
        return s

    df['speed'] = df.apply(speed, axis=1)
    return df[['id', 'BaseDateTime', 'LON', 'LAT', 'speed']]


def imp_df(files):
    large_df = []
    for i, file in enumerate(files):
        try:
            large_df.append(
                add_feat(
                    pd.read_csv(file,
                                names=colnames,
                                parse_dates=['BaseDateTime'],
                                engine='python')))
        except:
            print('error at: ' + file)
        print('%s df done' % os.path.splitext(file)[0])
    df = pd.concat(large_df)
    return df


def noise_filter(df):
    df = df[(df.LAT >= 39.6) & (df.LAT <= 40.2) & (df.LON >= 115.9) &
            (df.LON <= 116.8) & (df.speed < 60)]
    return df[['id', 'BaseDateTime', 'LON', 'LAT']]


def Liner_intrp(df, i):
    t = df.drop_duplicates(subset='BaseDateTime', keep='first')
    t = t.set_index('BaseDateTime')

    t0 = t
    t = t.resample('10min').asfreq()
    t = pd.concat([t, t0])
    del t0
    t = t.sort_values('BaseDateTime')

    t['id'] = t['id'].fillna(i)
    t['LON'] = t['LON'].interpolate(method='linear',
                                      limit_direction='forward',
                                      axis=0)
    t['LAT'] = t['LAT'].interpolate(method='linear',
                                    limit_direction='forward',
                                    axis=0)

    t = t.drop_duplicates()
    t = t.resample('10min').asfreq()
    t = t.dropna()

    df = t.reset_index()
    return df

def interpolation(df):
    """
        Doing data noise filtering.
        Make every 10 minutes existing a point.
    """
    df_a = pd.DataFrame()
    for idx in df.id.drop_duplicates():
        print('Doing interpolation for id:%d' % idx)
        t = df[df.id == idx]
        if t.shape == (3, ):
            df = df[~df.id.isin([idx])]
            continue
        df_a = pd.concat([df_a, Liner_intrp(t, idx)], axis=0)
        del t
    del df
    return df_a



def pre_file():
    if not os.path.exists('datas/file_pre.txt'):
        if not os.path.exists('datas/file.txt'):
            if os.path.exists('datas/file0.txt'):
                df = pd.read_csv('datas/file0.txt', parse_dates=['BaseDateTime'], engine='python')
            else:
                df = imp_df(read_file_all())
                write_tofile(df, 'file0')

            df = noise_filter(df)
            write_tofile(df, 'file')
            del df

        df = pd.read_csv('datas/file.txt', parse_dates=['BaseDateTime'], engine='python')
        df = interpolation(df)
        write_tofile(df, 'file_pre')

def pre_sub_file():
    if not os.path.exists('datas/sub_file_pre.txt'):
        if not os.path.exists('datas/sub_file.txt'):
            if os.path.exists('datas/sub_file0.txt'):
                df = pd.read_csv('datas/sub_file0.txt', parse_dates=['BaseDateTime'], engine='python')
            else:
                df = imp_df(read_file_sub())
                write_tofile(df, 'sub_file0')

            df = noise_filter(df)
            write_tofile(df, 'sub_file')
            del df

        df = pd.read_csv('datas/sub_file.txt', parse_dates=['BaseDateTime'], engine='python')
        df = interpolation(df)
        write_tofile(df, 'sub_file_pre')

if __name__ == "__main__":
    colnames = ['id', 'BaseDateTime', 'LON', 'LAT']
    pre_sub_file()
