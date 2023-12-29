# -*- coding: UTF-8 -*-

import math as m


def find_optimal_path(points, cur_path, cur_length, optimal_path, threshold):
    """
    points: 当前图中还未连接的点集
    cur_path: 当前形成的路径
    cur_length: 当前路径的长度
    optimal_path: 最优路径
    threshold: 最优路径的长度
    """

    if not points:
        print("cur_length: ", cur_length)
        print("threshold: ", threshold)
        if cur_length < threshold:
            for i in range(len(cur_path)):
                optimal_path[i] = cur_path[i][:]
            threshold = cur_length
            print("-------------------------")
            print(optimal_path)
            print(threshold)
        return threshold

    point_num = len(points)
    for i in range(point_num):
        cur_point = points.pop(i)
        tmp_len = 0
        if cur_path:
            tmp_len = m.sqrt(pow(cur_point[0] - cur_path[-1][0], 2) + pow(cur_point[1] - cur_path[-1][1], 2))
        cur_path.append(cur_point)

        threshold = find_optimal_path(points, cur_path, cur_length + tmp_len, optimal_path, threshold)

        cur_path.pop()
        points.insert(i, cur_point)

    return threshold


points = [[22, 23],
          [21, 4],
          [13, 13],
          [16, 6],
          [13, 23], ]
optimal_path = [0 for _ in range(len(points))]
threshold = 100000
threshold = find_optimal_path(points, [], 0, optimal_path, threshold)
print(optimal_path)
print(threshold)
