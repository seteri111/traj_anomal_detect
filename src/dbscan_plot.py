import numpy as np
import pandas as pd
import os.path

from matplotlib import pyplot as plt

from trajCluster.partition import approximate_trajectory_partitioning, segment_mdl_comp, rdp_trajectory_partitioning
from trajCluster.point import Point
from trajCluster.segment import Segment
from trajCluster.cluster import line_segment_clustering, representative_trajectory_generation

import time

def gather_tra():
    tra = dict()
    file_root = 'result/line_seg_files'
    # fn = "AIS_seg_"
    fn = "TD_seg_"
    for root, dir, files in os.walk(file_root):
        for f in files:
            if fn not in f:
                continue
            file_path = os.path.join(root, f)
            seg_t = pd.read_csv(file_path, engine='python')
            for idx in seg_t.traj_id.drop_duplicates():
                t = seg_t[seg_t.traj_id == idx]
                tra[idx] = []
                for k, s in t.iterrows():
                    tra[idx].append(Point(s.start_x, s.start_y))
                    tra[idx].append(Point(s.end_x, s.end_y))
            print('%s has imported!' % f)
    return tra


def read_cluster():
    norm_cluster = dict()
    cluster_long, cluster_lat = [], []
    file_root = 'result/AIS_cluster'
    for root, dir, files in os.walk(file_root):
        for f in files:
            file_path = os.path.join(root, f)
            clu_t = pd.read_csv(file_path, engine='python')
            for idx in clu_t.cluster_id.drop_duplicates():
                t = clu_t[clu_t.cluster_id == idx]
                if idx not in norm_cluster:
                    norm_cluster[idx] = []
                for k, s in t.iterrows():
                    norm_cluster[idx].extend([
                        Segment(Point(s.start_x, s.start_y),
                                Point(s.end_x, s.end_y), s.traj_id,
                                s.cluster_id)
                    ])
                    cluster_long.append(s.start_x)
                    cluster_long.append(s.end_x)
                    cluster_lat.append(s.start_y)
                    cluster_lat.append(s.end_y)
            print('%s has imported!' % f)
    return norm_cluster, cluster_long, cluster_lat

def cluster_group(norm_cluster):
    main_tra = representative_trajectory_generation(norm_cluster,
                                                    min_lines=5,
                                                    min_dist=0.4)
    return main_tra


def plot_tra(ax, tras, color, lw, alpha):
    for i, tra in tras.items():
        p_long = [p.x for p in tra]
        p_lat = [p.y for p in tra]
        ax.plot(p_long, p_lat, color, lw=lw, alpha=alpha)


def form_traj(tras):
    seg = []
    for idx, traj in tras.items():
        seg.append(
            pd.DataFrame(
                [[
                    traj[i].x, traj[i].y, traj[i + 1].x, traj[i + 1].y,
                    int(idx)
                ] for i in range(len(traj) - 1)],
                columns=['start_x', 'start_y', 'end_x', 'end_y',
                         'cluster_id']))
        print('Cluster %s has formed' % idx)
    seg = pd.concat(seg)
    return seg

def write_represent_traj(main_seg, tp):
    if main_seg.empty:
        print('error: there is no segment.')
        return

    file_root = 'result/represent_traj'
    f = 'main_seg_' + tp + '.txt'
    file = file_root + os.sep + f
    main_seg.to_csv(file, mode='w', index=False)


def main1():
    time_start = time.time()
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    tra = gather_tra()
    plot_tra(ax, tra, 'g-', 0.5, 0.2)
    del tra
    norm_cluster, cluster_long, cluster_lat = read_cluster()
    ax.scatter(cluster_long, cluster_lat, c='black', s=10)
    # plot_line_seg(ax, norm_cluster, 'b')
    del cluster_long, cluster_lat
    main_tra = cluster_group(norm_cluster)
    del norm_cluster
    plot_tra(ax, main_tra, 'r-', 1.5, 0.7)

    # plt.savefig("./line_seg_cluster/T_driver_csv_5_2.png", dpi=400)
    plt.savefig("result/line_seg_cluster/TD_csv_100_5_04_5_04h(2).png",
                dpi=600)
    time_end = time.time()
    print('totally cost', time_end - time_start)
    plt.show()

def main2():
    # tp = input('Input traj type')
    tp = 'TD'

    time_start = time.time()
    norm_cluster, cluster_long, cluster_lat = read_cluster()
    main_tra = cluster_group(norm_cluster)
    del norm_cluster, cluster_long, cluster_lat
    write_represent_traj(form_traj(main_tra), tp)

    time_end = time.time()
    print('totally cost', time_end - time_start)

if __name__ == "__main__":
    ch = input('Plot or Form?(1/2)')
    if ch == '1':
        main1()
    elif ch == '2':
        main2()