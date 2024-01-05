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

    path = [73, 61, 60, 59, 58, 72, 86, 85, 84, 83, 79, 80, 76, 71, 70, 69, 65, 62, 66, 67, 63, 50, 51, 52, 56, 57, 49,
            55, 47, 48, 33, 21, 34, 35, 22, 36, 37, 38, 23, 39, 44, 43, 24, 42, 41, 40, 11, 4, 10, 9, 3, 2, 8, 7, 6, 1,
            12, 5, 13, 18, 14, 15, 16, 17, 25, 19, 26, 27, 28, 20, 32, 31, 30, 29, 46, 54, 45, 53, 74, 64, 68, 75, 77,
            78, 81, 82, 87, 88, 92, 94, 99, 93, 89, 98, 100, 101, 102, 108, 107, 106, 105, 114, 112, 123, 130, 118, 121,
            124, 113, 125, 126, 131, 127, 128, 119, 115, 109, 110, 116, 120, 129, 122, 104, 103, 111, 117, 97, 91, 90,
            95, 96, 73]

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
