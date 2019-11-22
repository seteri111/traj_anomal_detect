# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------
# file      :cluster.py
# target    :轨迹之间聚类实现, 对所有的segment进行聚类, 使用dbscan密度聚类来实现segment的聚类
#
# output    :
# --------------------------------------------------------------------------------
import math
import numba
import numpy as np
import pandas as pd
from rtree import index

from .segment import compare, Segment
from .point import Point
from collections import deque, defaultdict

# from hausdorff import hausdorff_distance
import sys
sys.path.append("..")
# from discrete_frechet.frechetdist import frdist

import time

min_traj_cluster = 2  # 定义聚类的簇中至少需要的trajectory数量

@numba.jit(nopython=True, fastmath=True)
def haus_dist_lineseg(seg0, segs):
    """简易版近似hausdorff距离计算函数，特别计算一条线段对多条线段的hausdorff距离，矩阵运算，速度更快，但忽略了部分的垂直距离（会带来误差）
    parameter
    ---------
        seg0: nparray([start_x, start_y, end_x, end_y]), 表示一条线段
        segs: nparray([s_x1, s_y1, e_x1, e_y1], [...], ...)， 表示一组线段
    return
    ------
        dist: list[float], hausdorff距离的数组
    """
    start1 = seg0[:2]
    end1 = seg0[-2:]
    start2 = segs[..., :2]
    end2 = segs[..., -2:]

    dss = np.sqrt(np.sum((start2 - start1)**2, axis=1))
    dse = np.sqrt(np.sum((end2 - start1)**2, axis=1))
    des = np.sqrt(np.sum((start2 - end1)**2, axis=1))
    dee = np.sqrt(np.sum((end2 - end1) ** 2, axis=1))
    
    d1 = np.where((dss < des), dss, des)
    d2 = np.where((dse < dee), dse, dee)

    return np.where((d1 > d2), d1, d2)

    # sn = segs.shape[0]
    # dist = np.zeros(sn)
    # for i in range(sn):
    #     dist[i] = (hausdorff_distance(
    #         np.vstack((start1, end1)),
    #         np.vstack((start2[i], end2[i]))
    #     ))

    # return dist

@numba.jit(nopython=True, fastmath=True)
def get_bound(seg):
    xmin = min(seg[0], seg[2])
    xmax = max(seg[0], seg[2])
    ymin = min(seg[1], seg[3])
    ymax = max(seg[1], seg[3])
    return xmin, ymin, xmax, ymax


def build_rtree(segs):
    """构建segments的rtree索引
    parameter
    ---------
        segs: nparray([s_x1, s_y1, e_x1, e_y1], [...], ...)， 表示一组线段
    return
    ------
        idx: 所有line_segment的索引
    """
    idx = index.Index()

    for i in range(segs.shape[0]):
        idx.insert(i, (get_bound(segs[i])))

    return idx


def get_k_dist(segs, idx, k):
    """计算所有segment的k距离
    parameter 
    ---------
        segs: nparray([s_x1, s_y1, e_x1, e_y1], [...], ...), 所有的segment集合, 为所有集合的partition分段结果集合
        idx: R-Tree索引， 用于提高临近距离搜索速度
        k: 第k近的线段
    return
    ------
        k_dist： List[float, ...]， 返回全部seg第k近的距离大小
    """
    k_dist = []

    for seg in segs:
        idxs = list(idx.intersection((get_bound(seg))))

        dist = haus_dist_lineseg(seg, segs[idxs])
        dist.sort()
        if len(dist) >= k:
            k_dist.append(dist[k - 1])
        else:
            k_dist.append(dist[-1])
    return np.asfarray(k_dist)


def neighborhood(seg, segs, idx, epsilon=2.0):
    """计算一个segment在距离epsilon范围内的所有segment集合, 计算的时间复杂度为O(n). n为所有segment的数量
    parameter
    ---------
        seg: nparray([start_x, start_y, end_x, end_y]), 需要计算的segment对象
        segs: nparray([s_x1, s_y1, e_x1, e_y1], [...], ...)， 所有的segment集合
        idx: R-Tree索引， 用于提高临近距离搜索速度
        epsilon: float, segment之间的距离度量阈值
    return
    ------
        List[segment, ...], 返回seg在距离epsilon内的所有Segment集合.
    """
    xmin, ymin, xmax, ymax = get_bound(seg)
    xmax += epsilon
    xmin -= epsilon
    ymax += epsilon
    ymin -= epsilon

    idxs = list(idx.intersection((xmin, ymin, xmax, ymax)))

    # Calculate distance by hausdorff distance
    dist = haus_dist_lineseg(seg, segs[idxs])
    segment_set = np.array(idxs)[dist < epsilon]

    # seg_p = np.array([(seg.start_x, seg.start_y), (seg.end_x, seg.end_y)])

    # for i in idxs:
    #     if i == seg.name:
    #         continue

    #     segment_tmp = segs.loc[i]

        # Calculate distance by default
        # seg_long, seg_short = compare(
        #     seg, segment_tmp)  # get long segment by compare segment
        # if seg_long.get_all_distance(seg_short) <= epsilon:
        #     segment_set.append(segment_tmp)

        # Calculate distance by hausdorff distance
        # seg_p = np.array([(seg.start_x, seg.start_y), (seg.end_x, seg.end_y)])
        # segment_tmp_p = np.array([(segment_tmp.start_x, segment_tmp.start_y),
        #                           (segment_tmp.end_x, segment_tmp.end_y)])
        # if hausdorff_distance(seg_p, segment_tmp_p) <= epsilon:
        #     segment_set.append(i)

        # Calculate distance by frechet distance
        # seg_p = [[seg.start.x, seg.start.y], [seg.end.x, seg.end.y]]
        # segment_tmp_p = [[segment_tmp.start.x, segment_tmp.start.y],
        #                  [segment_tmp.end.x, segment_tmp.end.y]]
        # if frdist(seg_p, segment_tmp_p) <= epsilon:
        #     segment_set.append(segment_tmp)

    return segment_set


def expand_cluster(segs, idx, queue: deque, cluster_id: int, epsilon: float,
                   min_lines: int):
    while len(queue) != 0:
        curr_id = queue.popleft()
        curr_seg = segs[curr_id]
        curr_num_neighborhood = neighborhood(curr_seg[:4],
                                             segs[..., :4],
                                             idx,
                                             epsilon=epsilon)
        if len(curr_num_neighborhood) >= min_lines:
            m = curr_num_neighborhood[segs[curr_num_neighborhood, 4] == -1]
            segs[m, 4] = cluster_id
            for i in m:
                queue.append(i)


def line_segment_clustering(traj_segments,
                            epsilon: float = 2.0,
                            min_lines: int = 5):
    """线段segment聚类, 采用dbscan的聚类算法, 参考论文中的伪代码来实现聚类, 论文中的part4.2部分中的伪代码及相关定义
    parameter
    ---------
        traj_segments: pd.DataFrame(segs) 所有轨迹的partition划分后的segment集合.
        epsilon: float, segment之间的距离度量阈值
        min_lines: int or float, 轨迹在epsilon范围内的segment数量的最小阈值
    return
    ------
        Tuple[Dict[int, List[Segment, ...]], ...], 返回聚类的集合和不属于聚类的集合, 通过dict表示, key为cluster_id, value为segment集合
    """
    cluster_id = 0
    cluster_dict = defaultdict(list)

    segs = traj_segments[[
        'start_x', 'start_y', 'end_x', 'end_y', 'cluster_id'
    ]].values
    idx = build_rtree(segs[..., :4])
    print('R-Tree built!')

    for i in range(segs.shape[0]):
        # clu_start = time.time()
        seg = segs[i]

        # if i !=2 and i != 455 and i != 1661:
        #     continue      # a test of seg

        _queue = deque(list())
        if seg[4] == -1:
            seg_num_neighbor_set = neighborhood(seg[:4],
                                                segs[..., :4],
                                                idx,
                                                epsilon=epsilon)
            if len(seg_num_neighbor_set) >= min_lines:
                seg[4] = cluster_id
                segs[seg_num_neighbor_set, 4] = cluster_id
                for sub_seg in seg_num_neighbor_set:
                    # traj_segments.loc[
                    #     sub_seg,
                    #     'cluster_id'] = cluster_id  # assign clusterId to segment in neighborhood(seg)
                    _queue.append(sub_seg)  # insert sub segment into queue

                exp_start = time.time()
                expand_cluster(segs, idx, _queue, cluster_id, epsilon,
                               min_lines)
                exp_end = time.time()
                print('time for extend %d :' % cluster_id, exp_end - exp_start)

                cluster_id += 1
        # clu_end = time.time()
        # print('Calc tra %d :' % seg.traj_id, clu_end - clu_start)
        # print(seg.cluster_id, seg.traj_id)
        if seg[4] != -1:
            cluster_dict[seg[4]].append(i)  # 将轨迹放入到聚类的集合中, 按dict进行存放

    traj_segments.cluster_id = segs[..., 4]
    norm_cluster = []
    remove_cluster = dict()
    cluster_number = len(cluster_dict)
    for i in range(0, cluster_number):
        traj_num = len(
            set(map(lambda s: traj_segments.loc[s, 'traj_id'],
                    cluster_dict[i])))  # 计算每个簇下的轨迹数量
        print("the %d cluster lines:" % i, traj_num)
        if traj_num < min_traj_cluster:
            remove_cluster[i] = cluster_dict.pop(i)
        norm_cluster.append(traj_segments.loc[cluster_dict[i]])
    norm_cluster = pd.concat(norm_cluster)
    return norm_cluster, remove_cluster


def representative_trajectory_generation(cluster_segment: dict,
                                         min_lines: int = 3,
                                         min_dist: float = 2.0):
    """通过论文中的算法对轨迹进行变换, 提取代表性路径, 在实际应用中必须和当地的路网结合起来, 提取代表性路径, 该方法就是通过算法生成代表性轨迹
    parameter
    ---------
        cluster_segment: Dict[int, List[Segment, ...], ...], 轨迹聚类的结果存储字典, key为聚类ID, value为类簇下的segment列表
        min_lines: int, 满足segment数的最小值
        min_dist: float, 生成的轨迹点之间的最小距离, 生成的轨迹点之间的距离不能太近的控制参数
    return
    ------
        Dict[int, List[Point, ...], ...], 每个类别下的代表性轨迹结果
    """
    representive_point = defaultdict(list)
    for i in cluster_segment.keys():
        cluster_size = len(cluster_segment.get(i))
        sort_point = []  # [Point, ...], size = cluster_size*2
        rep_point, zero_point = Point(0, 0, -1), Point(1, 0, -1)

        # 对某个i类别下的segment进行循环, 计算类别下的平局方向向量: average direction vector
        for j in range(cluster_size):
            rep_point = rep_point + (cluster_segment[i][j].end -
                                     cluster_segment[i][j].start)
        rep_point = rep_point / float(cluster_size)  # 对所有点的x, y求平局值

        cos_theta = rep_point.dot(zero_point) / rep_point.distance(
            Point(0, 0, -1))  # cos(theta)
        sin_theta = math.sqrt(1 - math.pow(cos_theta, 2))  # sin(theta)

        # 对某个i类别下的所有segment进行循环, 每个点进行坐标变换: X' = A * X => X = A^(-1) * X'
        #   |x'|      | cos(theta)   sin(theta) |    | x |
        #   |  |  =   |                         | *  |   |
        #   |y'|      |-sin(theta)   cos(theta) |    | y |
        for j in range(cluster_size):
            s, e = cluster_segment[i][j].start, cluster_segment[i][j].end
            # 坐标轴变换后进行原有的segment修改
            cluster_segment[i][j] = Segment(
                Point(s.x * cos_theta + s.y * sin_theta,
                      s.y * cos_theta - s.x * sin_theta, -1),
                Point(e.x * cos_theta + e.y * sin_theta,
                      e.y * cos_theta - e.x * sin_theta, -1),
                traj_id=cluster_segment[i][j].traj_id,
                cluster_id=cluster_segment[i][j].cluster_id)
            sort_point.extend(
                [cluster_segment[i][j].start, cluster_segment[i][j].end])

        # 对所有点进行排序, 按照横轴的X进行排序, 排序后的point列表应用在后面的计算中
        sort_point = sorted(sort_point, key=lambda _p: _p.x)
        for p in range(len(sort_point)):
            intersect_cnt = 0.0
            start_y = Point(0, 0, -1)
            for q in range(cluster_size):
                s, e = cluster_segment[i][q].start, cluster_segment[i][q].end
                # 如果点在segment内则进行下一步的操作:
                if (sort_point[p].x <= e.x) and (sort_point[p].x >= s.x):
                    if s.x == e.x:
                        continue
                    elif s.y == e.y:
                        intersect_cnt += 1
                        start_y = start_y + Point(sort_point[p].x, s.y, -1)
                    else:
                        intersect_cnt += 1
                        start_y = start_y + Point(
                            sort_point[p].x, (e.y - s.y) / (e.x - s.x) *
                            (sort_point[p].x - s.x) + s.y, -1)
            # 计算the average coordinate: avg_p and dist >= min_dist
            if intersect_cnt >= min_lines:
                tmp_point: Point = start_y / intersect_cnt
                # 坐标转换到原始的坐标系, 通过逆矩阵的方式进行矩阵的计算:https://www.shuxuele.com/algebra/matrix-inverse.html
                tmp = Point(tmp_point.x * cos_theta - sin_theta * tmp_point.y,
                            sin_theta * tmp_point.x + cos_theta * tmp_point.y,
                            -1)
                _size = len(representive_point[i]) - 1
                if _size < 0 or (_size >= 0 and tmp.distance(
                        representive_point[i][_size]) > min_dist):
                    representive_point[i].append(tmp)
    return representive_point
