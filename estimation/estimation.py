# -*- coding: UTF-8 -*-
import math as m
import matplotlib.pyplot as plt

if __name__ == '__main__':
    with open('../dataset/pbn423.tsp', 'r') as file:
        lines = file.readlines()

    point_dict = dict()
    points = []
    lines = lines[8: -1]
    for line in lines:
        tmp_point = line.strip().split(' ')
        tmp_point = [float(x) for x in tmp_point]
        tmp_point[0] = int(tmp_point[0])
        point_dict[tmp_point[0]] = tuple(tmp_point[1:])
        points.append([tmp_point[1], tmp_point[2]])
    print(point_dict)

    points = [[77, 70],
              [78, 70],
              [79, 70],
              [80, 71],
              [78, 73],
              [79, 73],
              [80, 73]]

    # path = [13, 12, 14, 15, 5, 6, 7, 11, 9, 10, 8, 4, 2, 3, 1, 16, 13]
    #
    # dist = 0.0
    # print(len(path))
    # for i in range(len(path)):
    #     last_point = point_dict[path[i - 1]]
    #     cur_point = point_dict[path[i]]
    #     # last_point = path[i - 1]
    #     # cur_point = path[i]
    #     dist += round(m.sqrt(pow(last_point[0] - cur_point[0], 2) + pow(last_point[1] - cur_point[1], 2)))
    # print(dist)

    # x_values = [point_dict[path[i]][0] for i in range(len(path))]
    # y_values = [point_dict[path[i]][1] for i in range(len(path))]
    # # x_values = [path[i][0] for i in range(len(path))]
    # # y_values = [path[i][1] for i in range(len(path))]
    # plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', markersize=2, label='折线图')
    # plt.show()

    plt.xlim(76, 81)  # 设置 x 轴范围为 0 到 6
    plt.ylim(69, 74)  # 设置 y 轴范围为 0 到 8

    x_values = [points[i][0] for i in range(len(points))]
    y_values = [points[i][1] for i in range(len(points))]
    plt.scatter(x_values, y_values, marker='o', color='black', s=4)
    plt.scatter(78.7, 71.4, marker='x', color='blue', s=50)
    plt.show()

# if __name__ == '__main__':
#     with open('../dataset/rd100.tsp', 'r') as file:
#         lines = file.readlines()
#
#     point_dict = dict()
#     lines = lines[6: -1]
#     for line in lines:
#         tmp_point = line.strip().split(' ')
#         tmp_point = [float(x) for x in tmp_point]
#         tmp_point[0] = int(tmp_point[0])
#         point_dict[tmp_point[0]] = tuple(tmp_point[1:])
#     print(point_dict)
#
#     with open('rd100.opt.tour', 'r') as file:
#         lines = file.readlines()
#     lines = lines[4: -2]
#
#     path = []
#     for line in lines:
#         path.append(int(line))
#
#     dist = 0.0
#     for i in range(len(path)):
#         last_point = point_dict[path[i - 1]]
#         cur_point = point_dict[path[i]]
#         dist += round(m.sqrt(pow(last_point[0] - cur_point[0], 2) + pow(last_point[1] - cur_point[1], 2)))
#     print(dist)
