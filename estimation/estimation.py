# -*- coding: UTF-8 -*-
import math as m
import matplotlib.pyplot as plt

if __name__ == '__main__':
    with open('../dataset/wi29.tsp', 'r') as file:
        lines = file.readlines()

    point_dict = dict()
    lines = lines[7: -1]
    for line in lines:
        tmp_point = line.strip().split(' ')
        tmp_point = [float(x) for x in tmp_point]
        tmp_point[0] = int(tmp_point[0])
        point_dict[tmp_point[0]] = tuple(tmp_point[1:])
    print(point_dict)

    path = [11, 10, 6, 2, 1, 5, 4, 3, 7, 9, 8, 12, 13, 14, 20, 16, 24, 27, 25, 26, 28, 29, 21, 23, 22, 17, 18, 19, 15,
            11]

    dist = 0.0
    print(len(path))
    for i in range(len(path)):
        last_point = point_dict[path[i - 1]]
        cur_point = point_dict[path[i]]
        # last_point = path[i - 1]
        # cur_point = path[i]
        dist += round(m.sqrt(pow(last_point[0] - cur_point[0], 2) + pow(last_point[1] - cur_point[1], 2)))
    print(dist)

    x_values = [point_dict[path[i]][0] for i in range(len(path))]
    y_values = [point_dict[path[i]][1] for i in range(len(path))]
    # x_values = [path[i][0] for i in range(len(path))]
    # y_values = [path[i][1] for i in range(len(path))]
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', markersize=2, label='折线图')
    plt.show()

# if __name__ == '__main__':
#     with open('../dataset/pbn423.tsp', 'r') as file:
#         lines = file.readlines()
#
#     point_dict = dict()
#     lines = lines[8: -1]
#     for line in lines:
#         tmp_point = line.strip().split(' ')
#         tmp_point = [int(x) for x in tmp_point]
#         point_dict[tmp_point[0]] = tuple(tmp_point[1:])
#     print(point_dict)
#
#     with open('res_pbn423.tsp', 'r') as file:
#         lines = file.readlines()
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
