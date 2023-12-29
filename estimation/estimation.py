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

    path = [94, 92, 99, 93, 100, 98, 114, 118, 112, 123, 130, 121, 125, 126, 131, 127, 129, 120, 122, 117, 111, 103,
            104, 110, 116, 128, 119, 115, 109, 106, 105, 101, 102, 107, 124, 113, 108, 87, 89, 88, 81, 82, 75, 77, 78,
            68, 74, 54, 64, 65, 62, 69, 66, 70, 79, 76, 71, 67, 63, 50, 48, 47, 55, 49, 51, 57, 56, 52, 32, 31, 20, 30,
            29, 46, 28, 12, 5, 13, 18, 19, 25, 17, 14, 15, 16, 26, 53, 45, 27, 6, 1, 7, 11, 4, 10, 9, 3, 2, 8, 21, 33,
            34, 22, 35, 36, 37, 58, 38, 23, 39, 40, 41, 24, 42, 44, 43, 60, 61, 59, 73, 72, 80, 84, 83, 85, 86, 90, 91,
            97, 96, 95]
    dist = 0.0
    print(len(path))
    for i in range(len(path)):
        last_point = point_dict[path[i - 1]]
        cur_point = point_dict[path[i]]
        dist += round(m.sqrt(pow(last_point[0] - cur_point[0], 2) + pow(last_point[1] - cur_point[1], 2)))
    print(dist)

    x_values = [point_dict[path[i]][0] for i in range(len(path))]
    y_values = [point_dict[path[i]][1] for i in range(len(path))]
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
