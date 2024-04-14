# -*- coding: UTF-8 -*-
import math as m
import matplotlib.pyplot as plt
import numpy as np

from utils.read_dataset import read_dataset

# if __name__ == '__main__':
#     with open('../dataset/423/pbn423.tsp', 'r') as file:
#         lines = file.readlines()
#
#     point_dict = dict()
#     points = []
#     lines = lines[8: -1]
#     for line in lines:
#         tmp_point = line.strip().split(' ')
#         tmp_point = [float(x) for x in tmp_point]
#         tmp_point[0] = int(tmp_point[0])
#         point_dict[tmp_point[0]] = tuple(tmp_point[1:])
#         points.append([tmp_point[1], tmp_point[2]])
#     print(point_dict)
#
#     points = [[77, 70],
#               [78, 70],
#               [79, 70],
#               [80, 71],
#               [78, 73],
#               [79, 73],
#               [80, 73]]
#
#     path = [13, 12, 14, 15, 5, 6, 7, 11, 9, 10, 8, 4, 2, 3, 1, 16, 13]
#
#     dist = 0.0
#     print(len(path))
#     for i in range(len(path)):
#         last_point = point_dict[path[i - 1]]
#         cur_point = point_dict[path[i]]
#         # last_point = path[i - 1]
#         # cur_point = path[i]
#         dist += round(m.sqrt(pow(last_point[0] - cur_point[0], 2) + pow(last_point[1] - cur_point[1], 2)))
#     print(dist)
#
#     x_values = [point_dict[path[i]][0] for i in range(len(path))]
#     y_values = [point_dict[path[i]][1] for i in range(len(path))]
#     # x_values = [path[i][0] for i in range(len(path))]
#     # y_values = [path[i][1] for i in range(len(path))]
#     plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', markersize=2, label='折线图')
#     plt.show()

if __name__ == '__main__':
    lines = read_dataset('lin105.tsp', 105)
    point_dict = dict()
    for line in lines:
        tmp_point = line.strip().split(' ')
        tmp_point = [float(x) for x in tmp_point]
        tmp_point[0] = int(tmp_point[0])
        point_dict[tmp_point[0]] = tuple(tmp_point[1:])
    print(point_dict)

    lines = read_dataset('lin105.opt.tour', 105)

    path = []
    for line in lines:
        path.append(int(line))
    print(path)

    dist = 0.0
    for i in range(len(path)):
        last_point = point_dict[path[i - 1]]
        cur_point = point_dict[path[i]]
        dist += np.linalg.norm(np.array(last_point) - np.array(cur_point))
    print(dist)

    # x_values = [point_dict[path[i - 1]][0] for i in range(17)]
    # y_values = [point_dict[path[i - 1]][1] for i in range(17)]
    # plt.plot(x_values, y_values)
    # plt.show()
