import numpy as np
import pandas as pd
import os.path

from trajCluster.partition import approximate_trajectory_partitioning, segment_mdl_comp, rdp_trajectory_partitioning
from trajCluster.point import Point
from trajCluster.segment import Segment
from trajCluster.cluster import line_segment_clustering, representative_trajectory_generation

import time


def gather_line_seg():
    seg = []
    file_root = 'result/line_seg_files'
    fn = "AIS_seg_1_922"
    # fn = "TD_seg_"
    for root, dir, files in os.walk(file_root):
        for f in files:
            if fn not in f:
                continue
            file_path = os.path.join(root, f)
            seg_t = pd.read_csv(file_path, engine='python')
            seg.append(seg_t)
            print('%s has imported!' % f)
    seg = pd.concat(seg)
    return seg


def seg_DBSCAN(seg):
    """
    Do DBSCAN clustering for all line segments.
    """
    norm_cluster, remove_cluster = line_segment_clustering(seg,
                                                           min_lines=15,
                                                           epsilon=1.4)
    return norm_cluster

def write_cluster(clu):
    # if len(clu):
    #     print('error: there is no cluster.')
    #     return
    if clu.empty:
        print('error: there is no cluster.')
        return
    file_root = 'result/AIS_cluster'
    f = 'Cluster_AIS_922.txt'
    # f = 'Cluster_TD.txt'
    file = file_root + os.sep + f
    clu.to_csv(file, mode='w', index=False)


def main():
    time_start = time.time()

    seg = gather_line_seg()
    norm_cluster = seg_DBSCAN(seg)
    del seg
    write_cluster(norm_cluster)

    time_end = time.time()
    print('totally cost', time_end - time_start)

if __name__ == "__main__":
    main()