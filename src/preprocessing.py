import numpy as np
import pandas as pd
import os
import geopy.distance

# read


def read_data0():
    """
    Import datas from T-driver.
    Store them in 'file0.csv'.
    """
    res = []
    file_root = './release/taxi_log_2008_by_id'
    for root, dir, files in os.walk(file_root):
        for file in files:
            # testing in small number of data
            # if int(file.split('.')[0]) > 99:
            #     continue
            file_path = os.path.join(root, file)
            if os.path.getsize(file_path) == 0:
                continue
            df = pd.read_csv(file_path, header=None, engine='python')

            df.to_csv('file0.txt', mode='a', header=False, index=False)
            # df.to_csv('sub_file0.txt', mode='a', header=False, index=False)
            print('%s df done' % os.path.splitext(file)[0])
    del df
    res = pd.read_csv('file0.txt',
                      index_col=[0],
                      header=None,
                      names=['id', 'time', 'longitude', 'latitude'],
                      parse_dates=['time'],
                      engine='python')
    # res = pd.read_csv('sub_file0.txt', index_col=[0] , header=None,
    #                     names=['id', 'time', 'longitude', 'latitude'],
    #                     parse_dates=['time'], engine='python')
    return res


def write_tofile(df, s):
    df.to_csv(s + '.txt', mode='a')


# noise filtering


def Distance(lng1, lat1, lng2, lat2):
    """
        Caclulate distance between two points.
        inputs are longitude and latitude,
        output is distance(meter).
    """
    EARTH_RADIUS = 6378137
    radLon1 = lng1 * np.pi / 180
    radLat1 = lat1 * np.pi / 180
    radLon2 = lng2 * np.pi / 180
    radLat2 = lat2 * np.pi / 180
    s = np.arccos(
        np.sin(radLat1) * np.sin(radLat2) +
        np.cos(radLat1) * np.cos(radLat2) * np.cos(radLon1 - radLon2))
    s *= EARTH_RADIUS
    return s


def Velocity(data):
    """
        Caclulate the velocity(km/h) of data.
        Return a np.array.
    """
    l = data.shape[0]
    v = np.zeros(l)
    for i in range(l - 1):
        pt1 = data.iloc[i]
        pt2 = data.iloc[i + 1]
        timedelta = pt2.time - pt1.time
        t = timedelta.total_seconds()
        # s = Distance(pt1.longitude, pt1.latitude, pt2.longitude, pt2.latitude)
        # s = geopy.distance.vincenty((pt1.longitude, pt1.latitude),
        #                  (pt2.longitude, pt2.latitude)).meters  # a more accurate method, but will be removed
        s = geopy.distance.great_circle((pt1.longitude, pt1.latitude),
                         (pt2.longitude, pt2.latitude)).meters  # a better way using geopy tool
        if s == 0 or t == 0:
            continue
        v[i] = s / t
        v[i] *= 3.6  # transform m/s into km/h
    v[l - 1] = v[l - 2]
    return v


# print(distance(0, 100, 0, 100.1))


def noise_filter(df):
    """
        Doing data noise filtering.
        Latitude: from 39.7 to 39.9
        Longitude: from 116.1 to 116.9
        Velocity: lower than 60km/h
    """
    l0 = df.shape
    df = df[df['longitude'] >= 116.1]
    df = df[df['longitude'] <= 116.9]
    l1 = df.shape
    df = df[df['latitude'] >= 39.7]
    df = df[df['latitude'] <= 39.9]
    l2 = df.shape
    # df = df.drop_duplicates()
    # l3 = df.shape

    v = np.zeros(df.shape[0])
    j = 0
    for idx in df.index.drop_duplicates():
        print('Calculate velocity of id:%d' % idx)
        t = df.loc[idx]
        if t.shape == (3, ):
            v[j] = 100
            j += 1
            continue
        l = df.loc[idx].shape[0]
        v[j:j + l] = Velocity(t)
        j += l
    del j, t, l
    df.insert(3, 'velocity', v)
    l4 = df.shape
    # print(df.head(10))

    df = df[df['velocity'] < 60]
    l5 = df.shape
    print(l0)
    print(l1)
    print(l2)
    # print(l3)
    print(l4)
    print(l5)
    return df


# interpolation


def Liner_intrp(df, i):
    t = df.drop_duplicates(subset='time', keep='last')
    t = t.reset_index()
    t = t.set_index('time')

    # test
    # print(t)
    # print(t.index)
    # print(type(t['id']))

    t0 = t
    t = t.resample('10min').asfreq()
    t = pd.concat([t, t0])
    del t0
    t = t.sort_values('time')

    t['id'] = t['id'].fillna(i)
    t['longitude'] = t['longitude'].interpolate(method='linear',
                                                limit_direction='forward',
                                                axis=0)
    t['latitude'] = t['latitude'].interpolate(method='linear',
                                              limit_direction='forward',
                                              axis=0)

    # print(t)

    t = t.drop_duplicates()
    t = t.resample('10min').asfreq()
    t = t.dropna()

    # print(t)

    t = t.reset_index()
    df = t.set_index('id')
    # print(df)
    return df


def interpolation(df):
    """
        Doing data noise filtering.
        Make every 10 minutes existing a point.
    """
    df_a = pd.DataFrame()
    for idx in df.index.drop_duplicates():
        print('Doing interpolation for id:%d' % idx)
        t = df.loc[idx]
        if t.shape == (3, ):
            df.drop(idx, inplace=True)  # error 7789
            continue
        df_a = pd.concat([df_a, Liner_intrp(t, idx)], axis=0)
        del t
    del df
    return df_a


########################################################################################


def pre_file():
    if not os.path.exists('file_pre.txt'):
        if not os.path.exists('file.txt'):
            if os.path.exists('file0.txt'):
                df = pd.read_csv('file0.txt',
                                 index_col=[0],
                                 header=None,
                                 names=['id', 'time', 'longitude', 'latitude'],
                                 parse_dates=['time'],
                                 engine='python')
            else:
                df = read_data0()

            df = noise_filter(df)
            write_tofile(df, 'file')
            del df

        df = pd.read_csv('file.txt',
                         index_col=[0],
                         parse_dates=['time'],
                         usecols=[0, 1, 2, 3],
                         engine='python')
        df = interpolation(df)
        write_tofile(df, 'file_pre')


def pre_sub_file():
    if not os.path.exists('sub_file_pre.txt'):
        if not os.path.exists('sub_file.txt'):
            if os.path.exists('sub_file0.txt'):
                df = pd.read_csv('sub_file0.txt',
                                 index_col=[0],
                                 header=None,
                                 names=['id', 'time', 'longitude', 'latitude'],
                                 parse_dates=['time'],
                                 engine='python')
            else:
                df = read_data0()

            df = noise_filter(df)
            write_tofile(df, 'sub_file')
            del df

        df = pd.read_csv('sub_file.txt',
                         index_col=[0],
                         parse_dates=['time'],
                         usecols=[0, 1, 2, 3],
                         engine='python')
        df = interpolation(df)
        write_tofile(df, 'sub_file_pre')


# test
# print(df.index)
# print(df.head())
# print(df.info)
# print(df['time'].dtype, df['longitude'].dtype)
# print(df[df.index == 1])
# print(df.loc[10])
# print(df.loc['201', 'time'].head())
# print(df.head(30))
# print(df.head(30))
# print(df.head(30))
# print(df.iloc[0]['time'])
# print(df.loc['1'].shape)
# print(type(df.loc['1']))
# print(type(df.loc['1'].iloc[0]))
# print((df.loc['1'].iloc[1]['time'] - df.loc['1'].iloc[0]['time']).total_seconds())

# test
# print(df.head(100))
