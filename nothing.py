# -*- coding: UTF-8 -*-

import math as m
from cluster import SingleCluster


def cal_similarity(point1, point2):
    return m.sqrt(pow(point1[0] - point2[0], 2) + pow(point1[1] - point2[1], 2))


def find_optimal_path(points, cur_path, cur_len, opt_path, min_len, is_chosen):
    if len(cur_path) == (len(points) - 1):
        # 只剩下终点
        cur_len += cal_similarity(points[-1], points[cur_path[-1]])
        if cur_len < min_len:
            min_len = cur_len
            for i in range(len(cur_path)):
                opt_path[i] = cur_path[i]
            opt_path[-1] = len(points) - 1
        return min_len

    point_num = len(points)
    for i in range(1, point_num - 1):
        if is_chosen[i]:
            continue
        tmp_len = cal_similarity(points[i], points[cur_path[-1]])
        cur_path.append(i)
        is_chosen[i] = True

        min_len = find_optimal_path(points, cur_path, cur_len + tmp_len, opt_path, min_len, is_chosen)

        is_chosen[i] = False
        cur_path.pop()

    return min_len


# points = [[19, 41],
#           [29, 26],
#           [19, 29],
#           [33, 37],]
#           # [57, 87],
#           # [94, 63],
#           # [88, 88],
#           # [50, 11],
#           # [2, 30],
#           # [68, 83],
#           # [65, 14],
#           # [36, 29],
#           # [53, 99],]
#           # [94, 37],
#           # [6, 23],
#           # [72, 57],
#           # [24, 58],
#           # [45, 60],
#           # [59, 39],
#           # [97, 23]]
# cur_path = [0]
# optimal_path = [0 for _ in range(len(points))]
# threshold = 100000
# is_chosen = [False for _ in range(len(points))]
# threshold = find_optimal_path(points, cur_path, 0, optimal_path, threshold, is_chosen)
# print(optimal_path)
# print(threshold)


def find_diff_clusters_connector(cluster_1, cluster_2):
    points_1 = cluster_1.get_nodes_in_path()
    points_2 = cluster_2.get_nodes_in_path()
    conn_begin = 1 if (cluster_1.head == 0 and cluster_1.point_num > 1) else 0
    conn_end = 1 if (cluster_2.tail == 0 and cluster_2.point_num > 1) else 0
    conn_min_dist = cal_similarity(points_1[conn_begin], points_2[conn_end])

    for i in range(len(points_1)):
        if i == cluster_1.head:
            continue
        for j in range(len(points_2)):
            if j == cluster_2.tail:
                continue
            cur_dist = cal_similarity(points_1[i], points_2[j])
            if cur_dist < conn_min_dist:
                conn_min_dist = cur_dist
                conn_begin = i
                conn_end = j

    return conn_begin, conn_end


centroids = [[29.5, 42.75],
             [26.8, 29.8],
             [25.6, 23.0],
             [33.8, 29.0],
             [39.333333333333336, 34.833333333333336],
             [56.2, 45.6],
             [71.0, 46.0],
             [78.88888888888889, 36.77777777777778],
             [107.0, 27.0],
             [80.0, 28.5],
             [78.0, 20.4],
             [79.16666666666667, 7.833333333333333],
             [68.125, 13.5],
             [47.0, 21.25],
             [30.77777777777778, 13.222222222222221],
             [17.333333333333332, 19.666666666666668],
             [12.222222222222221, 8.555555555555555],
             [3.0, 8.5],
             [2.5, 24.25],
             [3.75, 37.5],
             [17.25, 32.5],
             [17.5, 42.333333333333336]]
points = [[(28, 40), (28, 43), (28, 47), (34, 41)],
          [(25, 28), (25, 29), (28, 28), (28, 30), (28, 34)],
          [(25, 22), (25, 23), (25, 24), (25, 26), (28, 20)],
          [(32, 26), (38, 30), (32, 31), (33, 26), (33, 29), (33, 31), (34, 26), (34, 29), (34, 31), (35, 31)],
          [(41, 32), (41, 34), (41, 35), (41, 36), (34, 38), (38, 34)],
          [(51, 45), (51, 47), (57, 44), (61, 45), (61, 47)],
          [(71, 45), (71, 47)],
          [(74, 35), (74, 39), (84, 34), (78, 35), (79, 33), (78, 39), (79, 37), (80, 41), (84, 38)],
          [(107, 27)],
          [(84, 24), (84, 29), (74, 29), (78, 32)],
          [(74, 24), (74, 20), (77, 21), (81, 17), (84, 20)],
          [(74, 6), (78, 10), (79, 10), (80, 10), (80, 5), (84, 6)],
          [(57, 12), (63, 6), (64, 22), (71, 11), (71, 13), (71, 16), (74, 12), (74, 16)],
          [(38, 20), (40, 22), (41, 23), (48, 22), (48, 27), (48, 6), (56, 25), (57, 25)],
          [(38, 16), (35, 17), (25, 11), (25, 15), (25, 9), (28, 16), (33, 15), (34, 15), (34, 5)],
          [(15, 19), (15, 25), (18, 17), (18, 19), (18, 21), (18, 23), (18, 25), (18, 15), (18, 13)],
          [(8, 0), (9, 10), (10, 10), (11, 10), (12, 10), (12, 5), (15, 13), (15, 8), (18, 11)],
          [(0, 13), (2, 0), (5, 13), (5, 8)],
          [(0, 27), (5, 19), (0, 26), (5, 25)],
          [(0, 39), (5, 31), (5, 37), (5, 43)],
          [(15, 31), (15, 37), (18, 29), (18, 31), (18, 33), (18, 35), (18, 37), (18, 27)],
          [(15, 43), (18, 39), (18, 41), (18, 42), (18, 44), (18, 45)]]
path = []
for i in range(len(centroids)):
    path.append(SingleCluster(centroids[i], points[i]))
for i in range(len(path)):
    path[i - 1].tail, path[i].head = find_diff_clusters_connector(path[i - 1], path[i])
    if i > 0:
        path[i - 1].determine_head_and_tail()
path[len(path) - 1].determine_head_and_tail()
for i in range(len(path)):
    path[i].find_optimal_path(27)

# 还原到单个节点状态
for i in range(len(path) - 1, -1, -1):
    path[i: i + 1] = path[i].points

print(path)
