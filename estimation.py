# -*- coding: UTF-8 -*-
import math as m
import matplotlib.pyplot as plt

if __name__ == '__main__':
    with open('dataset/xqf131.tsp', 'r') as file:
        lines = file.readlines()

    point_dict = dict()
    lines = lines[8: -1]
    for line in lines:
        tmp_point = line.strip().split(' ')
        tmp_point = [int(x) for x in tmp_point]
        point_dict[tmp_point[0]] = tuple(tmp_point[1:])
    print(point_dict)

    path = [32, 33, 35, 34, 47, 49, 50, 51, 56, 57, 52, 36, 22, 37, 38, 39, 23, 11, 10, 24, 40, 41, 42, 44, 43, 60, 59,
            61, 104, 111, 110, 117, 122, 103, 96, 97, 91, 90, 95, 92, 108, 113, 126, 125, 124, 121, 130, 123, 118, 114,
            105, 112, 98, 100, 101, 102, 106, 99, 107, 131, 115, 109, 127, 119, 116, 120, 129, 128, 94, 93, 88, 87, 89,
            69, 65, 62, 66, 79, 76, 71, 67, 63, 70, 58, 72, 73, 80, 84, 85, 86, 83, 82, 81, 78, 77, 75, 68, 64, 48, 55,
            54, 46, 74, 53, 45, 26, 27, 19, 14, 15, 16, 17, 18, 25, 12, 13, 5, 1, 6, 7, 8, 2, 3, 4, 9, 21, 20, 29, 28,
            30, 31]
    dist = 0.0
    print(len(path))
    for i in range(len(path)):
        last_point = point_dict[path[i - 1]]
        cur_point = point_dict[path[i]]
        dist += round(m.sqrt(pow(last_point[0] - cur_point[0], 2) + pow(last_point[1] - cur_point[1], 2)))
    print(dist)
    # 828
    # 0.468

    x_values = [point_dict[path[i]][0] for i in range(len(path))]
    y_values = [point_dict[path[i]][1] for i in range(len(path))]
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', markersize=2, label='折线图')
    plt.show()


# if __name__ == '__main__':
#     with open('dataset/xqf131.tsp', 'r') as file:
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
#     with open('dataset/res_xqf131.tsp', 'r') as file:
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
#     # 564
