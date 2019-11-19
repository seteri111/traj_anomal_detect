import os

import numpy as np
import pandas as pd
from collections import Counter

from trajCluster.cluster import build_rtree, get_k_dist

import time
from matplotlib import pyplot as plt

def gather_line_seg(s, e):
    segs = []
    file_root = 'result/line_seg_files'
    # fn = "AIS_seg_" + s + "_" + e
    fn = "TD_seg_"
    for root, dir, files in os.walk(file_root):
        for f in files:
            if fn not in f:
                continue
            file_path = os.path.join(root, f)
            seg_t = pd.read_csv(file_path, engine='python')
            segs.append(seg_t)
            print('%s has imported!' % f)
    segs = pd.concat(segs)
    return segs[['start_x', 'start_y', 'end_x', 'end_y']].values

def identify_par(segs, k, ax):
    """确定DBSCAN最适参数，绘制k_dist变化趋势
    """
    idx = build_rtree(segs)
    print('R-Tree built!')
    k_dist = get_k_dist(segs, idx, k)
    bins = np.arange(0, 3, 0.03)
    
    ax.hist(k_dist, bins, color='b', alpha=0.4)
    

def main():
    k = int(input('Input min_lines:'))

    time_start = time.time()
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)

    segs = gather_line_seg('1', '235000000')
    identify_par(segs, k, ax)

    plt.xlabel('k_dist')
    plt.ylabel('count')

    time_end = time.time()
    print('totally cost', time_end - time_start)
    plt.show()

if __name__ == "__main__":
    main()