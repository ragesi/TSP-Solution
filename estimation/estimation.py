# -*- coding: UTF-8 -*-
import math as m
import matplotlib.pyplot as plt

if __name__ == '__main__':
    with open('../dataset/xqg237.tsp', 'r') as file:
        lines = file.readlines()

    point_dict = dict()
    lines = lines[8: -1]
    for line in lines:
        tmp_point = line.strip().split(' ')
        tmp_point = [float(x) for x in tmp_point]
        tmp_point[0] = int(tmp_point[0])
        point_dict[tmp_point[0]] = tuple(tmp_point[1:])
    print(point_dict)

    path = [187, 188, 180, 179, 178, 171, 142, 143, 144, 134, 133, 125, 126, 127, 112, 114, 113, 124, 123, 122, 132,
            131, 141, 164, 163, 162, 161, 160, 159, 158, 186, 170, 169, 168, 177, 176, 175, 153, 154, 155, 156, 157,
            140, 139, 138, 120, 121, 109, 110, 102, 111, 108, 100, 99, 84, 85, 86, 76, 75, 68, 67, 62, 51, 43, 44, 45,
            54, 53, 52, 63, 64, 69, 70, 71, 72, 77, 78, 73, 66, 46, 28, 25, 13, 4, 3, 7, 12, 17, 22, 21, 11, 10, 20, 35,
            2, 6, 5, 1, 9, 8, 19, 14, 15, 16, 18, 23, 24, 29, 31, 26, 32, 33, 34, 27, 30, 42, 50, 49, 41, 40, 39, 38,
            37, 36, 55, 47, 48, 57, 61, 60, 56, 58, 59, 65, 87, 88, 89, 101, 96, 93, 79, 74, 80, 81, 103, 97, 104, 105,
            106, 94, 90, 82, 95, 91, 83, 92, 98, 107, 119, 118, 130, 129, 117, 116, 115, 128, 135, 145, 146, 147, 136,
            148, 149, 150, 137, 151, 152, 167, 166, 174, 165, 172, 173, 181, 182, 189, 198, 190, 191, 183, 214, 222,
            225, 227, 226, 203, 204, 205, 192, 184, 185, 193, 194, 197, 196, 195, 215, 228, 229, 230, 231, 219, 216,
            206, 207, 208, 209, 210, 218, 217, 220, 232, 233, 221, 223, 234, 235, 237, 236, 224, 211, 212, 202, 213,
            201, 200, 199, 187]

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
