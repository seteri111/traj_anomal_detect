from preprocessing2 import pre_file, pre_sub_file

import numpy as np
import pandas as pd
import os.path
import sys

from trajCluster.partition import approximate_trajectory_partitioning, segment_mdl_comp, rdp_trajectory_partitioning
from trajCluster.point import Point
from trajCluster.segment import Segment
from trajCluster.cluster import line_segment_clustering, representative_trajectory_generation

import time


def get_T_Drive_data():
    try:
        df = pd.read_csv('datas/sub_file.txt',
                         parse_dates=['BaseDateTime'],
                         engine='python')
    except:
        pre_sub_file()
        df = pd.read_csv('datas/sub_file.txt',
                         parse_dates=['BaseDateTime'],
                         engine='python')
    return df


def get_AIS_data():
    n = input('Input month length:')
    f = 'datas/AIS_cvs_file_' + str(n) + '.txt'
    try:
        df = pd.read_csv(f,
                         usecols=[0, 1, 2, 3],
                         parse_dates=['BaseDateTime'],
                         engine='python')
        print('%s written!' % os.path.splitext(f)[0])
    except:
        print('error at: ' + f)
    return df.sort_values('id')


def get_AIS_partition():
    s = input('Input start file:')
    e = input('Input end file:')
    n = input('Input month length:')
    root = 'result/AIS_cell'
    large_df = []
    for i in range(int(s), int(e) + 1):
        f = 'AIS_' + n + '_part_' + str(i) + '.txt'
        file_path = root + os.sep + f
        if not os.path.exists(file_path):
            continue
        try:
            large_df.append(
                pd.read_csv(file_path,
                            usecols=[0, 1, 2, 3],
                            parse_dates=['BaseDateTime'],
                            engine='python'))
            print('%s written!' % os.path.splitext(f)[0])
        except:
            print('error at: ' + f)
    df = pd.concat(large_df)
    return df.sort_values('id'), s, e


def get_tra_part_T(df, start, end):
    """
    Get every single trajectory from dataset by id.
    And then get group of their segmentation.
    """

    tra = dict()
    line_seg = dict()
    seg = []

    for idx in df.id.drop_duplicates():
        if idx < start or idx > end:
            continue
        t = df[df.id == idx]
        t = t.reset_index()
        t.sort_values('BaseDateTime')
        tra[idx] = [
            Point(t.LON[i] * 100, t.LAT[i] * 100)
            for i in range(0, t.shape[0])
        ]
        line_seg = approximate_trajectory_partitioning(tra[idx],
                                                       theta=0,
                                                       traj_id=int(idx))
        # line_seg = [
        #     Segment(Point(t.LON[i] * 10, t.LAT[i] * 10),
        #             Point(t.LON[i + 1] * 10, t.LAT[i + 1] * 10), int(idx))
        #     for i in range(0, t.shape[0] - 1)
        # ]
        seg.append(
            pd.DataFrame([[
                line_seg[i].start.x, line_seg[i].start.y, line_seg[i].end.x,
                line_seg[i].end.y, line_seg[i].traj_id, line_seg[i].cluster_id
            ] for i in range(0, len(line_seg))],
                         columns=[
                             'start_x', 'start_y', 'end_x', 'end_y', 'traj_id',
                             'cluster_id'
                         ]))
        print('Segment of trajectory %s has done' % idx)
    seg = pd.concat(seg)
    return seg


def get_tra_seg_A(df, end):
    """
    Get every single trajectory from dataset by id.
    And then get group of their segmentation.
    """

    line_seg = dict()
    seg = []

    for idx in df.id.drop_duplicates():
        if idx > end:
            continue
        t = df[df.id == idx]
        t = t.reset_index()
        t.sort_values('BaseDateTime')
        line_seg = [
            Segment(Point(t.LON[i] * 1, t.LAT[i] * 1),
                    Point(t.LON[i + 1] * 1, t.LAT[i + 1] * 1), int(idx))
            for i in range(0, t.shape[0] - 1)
        ]
        seg.append(
            pd.DataFrame([[
                line_seg[i].start.x, line_seg[i].start.y, line_seg[i].end.x,
                line_seg[i].end.y, line_seg[i].traj_id, line_seg[i].cluster_id
            ] for i in range(0, len(line_seg))],
                         columns=[
                             'start_x', 'start_y', 'end_x', 'end_y', 'traj_id',
                             'cluster_id'
                         ]))
        print('Segment of trajectory %s has done' % idx)
    seg = pd.concat(seg)
    return seg


def write2fileT(seg, start, end):
    dir = "result/line_seg_files"
    f = "TD_seg_" + str(start) + "_" + str(end) + ".txt"
    file = dir + os.sep + f
    seg.to_csv(file, mode='w', index=False)


def write2fileA(seg, s, e):
    dir = "result/line_seg_files"
    f = "AIS_seg_" + s + "_" + e + ".txt"
    file = dir + os.sep + f
    seg.to_csv(file, mode='w', index=False)

if __name__ == "__main__":
    # start = int(input("Please input first trajectory:"))
    # end = int(input("Please input last trajectory:"))

    time_start = time.time()

    # write2fileT(get_tra_part_T(get_T_Drive_data(), start, end), start, end)

    # end = sys.maxsize
    end = 235000000
    df = get_AIS_data()
    write2fileA(get_tra_seg_A(df, end), '1', str(end))

    time_end = time.time()
    print('totally cost', time_end - time_start)

