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

    path = [82, 87, 88, 92, 94, 99, 89, 93, 98, 100, 101, 102, 105, 118, 123, 130, 121, 114, 112, 124, 113, 108, 106,
            107, 125, 126, 127, 131, 129, 128, 119, 115, 109, 110, 116, 120, 117, 122, 111, 104, 103, 96, 97, 95, 91,
            90, 86, 85, 84, 83, 80, 79, 76, 67, 66, 70, 65, 69, 71, 63, 51, 55, 47, 48, 49, 50, 52, 56, 62, 57, 58, 72,
            73, 61, 60, 59, 40, 24, 44, 43, 42, 41, 39, 23, 38, 37, 36, 35, 22, 9, 10, 11, 4, 3, 2, 7, 8, 21, 34, 33,
            32, 31, 20, 30, 29, 28, 15, 6, 1, 12, 14, 5, 13, 18, 25, 17, 16, 19, 27, 26, 45, 53, 74, 46, 54, 64, 68, 75,
            77, 78, 81, 82]

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
