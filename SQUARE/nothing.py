# -*- coding: UTF-8 -*-

import math as m

# import sys
# sys.path.append("..")
from utils.read_dataset import read_dataset


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


lines = read_dataset('ulysses16.tsp', 16)
points = []
for line in lines:
    point = line.strip().split(' ')
    point = [int(point[0]), float(point[1]), float(point[2])]
    points.append((point[1], point[2]))
points.append(points[0])
print(len(points))

cur_path = [0]
optimal_path = [0 for _ in range(len(points))]
threshold = 100000
is_chosen = [False for _ in range(len(points))]
threshold = find_optimal_path(points, cur_path, 0, optimal_path, threshold, is_chosen)

# x_values = [cycle_test_for_4[optimal_path[i]][0] for i in range(len(optimal_path))]
# y_values = [cycle_test_for_4[optimal_path[i]][1] for i in range(len(optimal_path))]
# plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', markersize=2, label='折线图')
# plt.show()

print(optimal_path)
print(threshold)


def find_diff_clusters_connector(cluster_1, cluster_2):
    points_1 = cluster_1.get_nodes_in_path()
    points_2 = cluster_2.get_nodes_in_path()
    conn_begin = 1 if (cluster_1.head == 0 and cluster_1.element_num > 1) else 0
    conn_end = 1 if (cluster_2.tail == 0 and cluster_2.element_num > 1) else 0
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


centroids = [[2.857142857142857, 32.57142857142857],
             [3.75, 13.25],
             [8.2, 5.0],
             [14.833333333333334, 10.833333333333334],
             [17.4, 18.2],
             [17.4, 25.8],
             [26.2, 28.2],
             [25.75, 22.25],
             [28.333333333333332, 11.833333333333334],
             [37.666666666666664, 18.833333333333332],
             [54.0, 19.857142857142858],
             [71.5, 14.75],
             [79.16666666666667, 7.833333333333333],
             [85.85714285714286, 24.571428571428573],
             [77.0, 33.5],
             [76.33333333333333, 41.5],
             [56.2, 45.6],
             [40.0, 33.5],
             [33.333333333333336, 28.88888888888889],
             [30.0, 40.5],
             [17.5, 42.333333333333336],
             [17.0, 34.0]]
points = [[(0, 26), (0, 27), (0, 39), (5, 25), (5, 31), (5, 37), (5, 43)],
          [(0, 13), (5, 13), (5, 19), (5, 8)],
          [(2, 0), (8, 0), (9, 10), (10, 10), (12, 5)],
          [(11, 10), (12, 10), (15, 13), (15, 8), (18, 11), (18, 13)],
          [(15, 19), (18, 15), (18, 17), (18, 19), (18, 21)],
          [(15, 25), (18, 23), (18, 25), (18, 27), (18, 29)],
          [(25, 26), (25, 28), (25, 29), (28, 28), (28, 30)],
          [(25, 22), (25, 23), (25, 24), (28, 20)],
          [(25, 11), (25, 15), (25, 9), (28, 16), (33, 15), (34, 5)],
          [(34, 15), (35, 17), (38, 16), (38, 20), (40, 22), (41, 23)],
          [(48, 22), (48, 27), (48, 6), (56, 25), (57, 12), (57, 25), (64, 22)],
          [(63, 6), (71, 11), (71, 13), (71, 16), (74, 12), (74, 16), (74, 20), (74, 24)],
          [(74, 6), (78, 10), (79, 10), (80, 10), (80, 5), (84, 6)],
          [(77, 21), (81, 17), (84, 20), (84, 24), (84, 29), (84, 34), (107, 27)],
          [(74, 29), (74, 35), (78, 32), (78, 35), (79, 33), (79, 37)],
          [(71, 45), (71, 47), (74, 39), (78, 39), (80, 41), (84, 38)],
          [(51, 45), (51, 47), (57, 44), (61, 45), (61, 47)],
          [(38, 30), (38, 34), (41, 32), (41, 34), (41, 35), (41, 36)],
          [(32, 26), (32, 31), (33, 26), (33, 29), (33, 31), (34, 26), (34, 29), (34, 31), (35, 31)],
          [(28, 34), (28, 40), (28, 43), (28, 47), (34, 38), (34, 41)],
          [(15, 43), (18, 39), (18, 41), (18, 42), (18, 44), (18, 45)],
          [(15, 31), (15, 37), (18, 31), (18, 33), (18, 35), (18, 37)]]
# path = []
# for i in range(len(centroids)):
#     path.append(SingleCluster(centroids[i], points[i]))
# for i in range(len(path)):
#     path[i - 1].tail, path[i].head = find_diff_clusters_connector(path[i - 1], path[i])
#     if i > 0:
#         path[i - 1].determine_head_and_tail()
# path[len(path) - 1].determine_head_and_tail()
# for i in range(len(path)):
#     path[i].find_optimal_path(27)
#
# # 还原到单个节点状态
# for i in range(len(path) - 1, -1, -1):
#     path[i: i + 1] = path[i].elements
#
# print(path)
