import numpy as np
import pandas as pd
import os.path

from matplotlib import pyplot as plt

from trajCluster.partition import approximate_trajectory_partitioning, segment_mdl_comp, rdp_trajectory_partitioning
from trajCluster.point import Point
from trajCluster.segment import Segment
from trajCluster.cluster import line_segment_clustering, representative_trajectory_generation

import time

time_start=time.time()

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
                                                    min_dist=0.7)
    return main_tra


def plot_tra(ax, tras, color, lw, alpha):
    for i, tra in tras.items():
        p_long = [p.x for p in tra]
        p_lat = [p.y for p in tra]
        ax.plot(p_long, p_lat, color, lw=lw, alpha=alpha)

def main():
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
    plt.savefig("result/line_seg_cluster/TD_csv_100_5_04_5_07.png", dpi=600)
    time_end=time.time()
    print('totally cost', time_end - time_start)
    plt.show()

if __name__ == "__main__":
    main()