import os.path
import numpy as np
import pandas as pd

def read_file_all():
    large_df = []
    file_root = 'datas/AIS_Data_txt'
    for root, dir, files in os.walk(file_root):
        for f in files:
            if os.path.splitext(f)[1] == '.csv':
                if 'Zone03' not in f:
                    continue
                try:
                    file_path = os.path.join(root, f)
                    large_df.append(
                            pd.read_csv(file_path,
                                        usecols=[0, 1, 2, 3, 4, 5],
                                        parse_dates=['BaseDateTime'],
                                        engine='python'))
                    print('%s written!' % os.path.splitext(f)[0])
                except:
                    print('error at: ' + f)
    df = pd.concat(large_df)
    df.rename(columns={'MMSI':'id'},inplace=True)
    return df.sort_values('id')

def write_tofile(df, s):
    fileroot = 'datas'
    f = fileroot + os.sep + s + '.txt'
    df.to_csv(f, mode='w', index=False)

def get_csv_data():
    write_tofile(read_file_all(), 'AIS_cvs_file_1')

if __name__ == "__main__":
    get_csv_data()
