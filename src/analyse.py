import numpy as np
import pandas as pd

from Preprocessing2 import pre_file, pre_sub_file
from import_csv import get_csv_data

from trajCluster.partition import approximate_trajectory_partitioning, segment_mdl_comp, rdp_trajectory_partitioning
from trajCluster.point import Point
from trajCluster.cluster import line_segment_clustering, representative_trajectory_generation

from matplotlib import pyplot as plt

def get_sub_data():
    try:
        df = pd.read_csv('sub_file_pre.txt', parse_dates=['BaseDateTime'], engine='python')
    except:
        pre_sub_file()
        df = pd.read_csv('sub_file_pre.txt', parse_dates=['BaseDateTime'], engine='python')
    return df

def get_full_data():
    try:
        df = pd.read_csv('file_pre.txt', parse_dates=['BaseDateTime'], engine='python')
    except:
        pre_file()
        df = pd.read_csv('file_pre.txt', parse_dates=['BaseDateTime'], engine='python')
    return df

def get_AIS_csv():
    try:
        df = pd.read_csv('AIS_cvs_file.txt', usecols=[0, 1, 2, 3], parse_dates=['BaseDateTime'], engine='python')
    except:
        get_csv_data()
        df = pd.read_csv('AIS_cvs_file.txt', usecols=[0, 1, 2, 3], parse_dates=['BaseDateTime'], engine='python')
    return df

def get_tra_part(df):
    """
    Get every single trajectory from dataset by id.
    And then get group of their segmentation.
    """

    tra = dict()
    line_seg = dict()
    seg = []
    for idx in df.id.drop_duplicates():
        # if idx > 20:
        #     continue
        t = df[df.id == idx]
        t = t.reset_index()
        t.sort_values('BaseDateTime')
        tra[idx] = [Point(t.LON[i], t.LAT[i]) for i in range(0, t.shape[0])]
        line_seg[idx] = approximate_trajectory_partitioning(tra[idx], theta=0, traj_id=int(idx))
        seg.extend(line_seg[idx])
        print('Segment of trajectory %s has done' % idx)
    return tra, line_seg, seg

def seg_DBSCAN(seg):
    """
    Do DBSCAN clustering for all line segments.
    """
    norm_cluster, remove_cluster = line_segment_clustering(seg, min_lines=3, epsilon=15.0)
    cluster_long, cluster_lat = [], []
    for k, v in norm_cluster.items():
        cluster_long.extend([s.start.x for s in v])
        cluster_long.extend([s.end.x for s in v])

        cluster_lat.extend([s.start.y for s in v])
        cluster_lat.extend([s.end.y for s in v])
        print("using cluster: the cluster %d, the segment number %d" % (k, len(v)))

    return norm_cluster, cluster_long, cluster_lat

def cluster_group(norm_cluster):
    main_tra = representative_trajectory_generation(norm_cluster, min_lines=2, min_dist=1.0)
    return main_tra

def plot_tra(ax, tras, color):
    for i, tra in tras.items():
        p_long = [p.x for p in tra]
        p_lat = [p.y for p in tra]
        ax.plot(p_long, p_lat, color, lw=2.0)


# df = get_sub_data()
df = get_AIS_csv()
tra, line_seg, seg = get_tra_part(df)

del df
norm_cluster, cluster_long, cluster_lat = seg_DBSCAN(seg)

main_tra = cluster_group(norm_cluster)

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111)
plot_tra(ax, tra, 'g-')
plot_tra(ax, main_tra, 'r-')
ax.scatter(cluster_long, cluster_lat, c='y')

# plt.savefig("T_Diver_cluster_sub.png", dpi=400)
plt.savefig("AIS_csv.png", dpi=400)
plt.show()