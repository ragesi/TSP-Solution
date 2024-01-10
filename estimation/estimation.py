# -*- coding: UTF-8 -*-
import math as m
import matplotlib.pyplot as plt

if __name__ == '__main__':
    with open('../dataset/xqf131.tsp', 'r') as file:
        lines = file.readlines()

    point_dict = dict()
    lines = lines[8: -1]
    for line in lines:
        tmp_point = line.strip().split(' ')
        tmp_point = [int(x) for x in tmp_point]
        point_dict[tmp_point[0]] = tuple(tmp_point[1:])
    print(point_dict)

    path = [10, 11, 4, 9, 3, 2, 8, 7, 1, 6, 12, 5, 13, 18, 14, 15, 16, 17, 25, 19, 26, 27, 28, 29, 20, 30, 31, 32, 33,
            21, 35, 34, 49, 55, 46, 47, 48, 50, 51, 52, 56, 57, 63, 67, 71, 76, 70, 66, 65, 69, 62, 54, 45, 53, 74, 89,
            64, 68, 75, 77, 78, 81, 82, 87, 88, 92, 94, 93, 99, 107, 108, 106, 101, 100, 98, 102, 105, 114, 112, 123,
            130, 118, 121, 124, 113, 125, 126, 127, 131, 129, 128, 119, 115, 109, 110, 116, 120, 117, 122, 111, 103,
            104, 96, 97, 95, 91, 90, 86, 85, 84, 83, 79, 80, 72, 73, 61, 60, 58, 59, 40, 24, 44, 43, 42, 41, 39, 38, 37,
            36, 22, 23, 10]

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
