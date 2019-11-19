import numpy as np
import pandas as pd
import os.path

from trajCluster.partition import approximate_trajectory_partitioning, segment_mdl_comp, rdp_trajectory_partitioning
from trajCluster.point import Point
from trajCluster.segment import Segment
from trajCluster.cluster import line_segment_clustering, representative_trajectory_generation

import time


def gather_line_seg(s, e):
    seg = []
    file_root = 'result/line_seg_files'
    # fn = "AIS_seg_" + s + "_" + e
    fn = "TD_seg_"
    for root, dir, files in os.walk(file_root):
        for f in files:
            if fn not in f:
                continue
            file_path = os.path.join(root, f)
            seg_t = pd.read_csv(file_path, engine='python', index_col='seg_id')
            seg.append(seg_t)
            print('%s has imported!' % f)
    seg = pd.concat(seg)
    return seg


def seg_DBSCAN(seg):
    """
    Do DBSCAN clustering for all line segments.
    """
    norm_cluster, remove_cluster = line_segment_clustering(seg,
                                                           min_lines=6,
                                                           epsilon=0.4)
    return norm_cluster

def write_cluster(clu, s, e):
    # if len(clu):
    #     print('error: there is no cluster.')
    #     return
    if clu.empty:
        print('error: there is no cluster.')
        return
    file_root = 'result/AIS_cluster'
    f = 'Cluster_' + s + '_' + e + '.txt'
    file = file_root + os.sep + f
    clu.to_csv(file, mode='w', index=False)


def main():
    s = input("Input start:")
    e = input("Input end:")

    time_start = time.time()

    seg = gather_line_seg(s, e)
    norm_cluster = seg_DBSCAN(seg)
    del seg
    write_cluster(norm_cluster, s, e)

    time_end = time.time()
    print('totally cost', time_end - time_start)

if __name__ == "__main__":
    main()