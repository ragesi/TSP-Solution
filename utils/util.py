# -*- coding: UTF-8 -*-
import numpy as np


def decimal_to_binary(real, qubit_num):
    """
    change decimal to binary
    :param real: the decimal that waiting to be changed, which must be less than 1
    :param qubit_num: the number of binary bits
    :return: the binary string
    """
    if real >= 1.0:
        return ""
    res = ""
    for i in np.arange(qubit_num):
        cur_num = 1.0 / (2 ** (i + 1))
        if real >= cur_num:
            res += '1'
            real -= cur_num
        else:
            res += '0'
    return res


def int_to_binary(val, qubit_num):
    res = bin(val)[2:]
    res = '0' * (qubit_num - len(res)) + res
    return res


def cal_similarity(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


def find_optimal_path(points, cur_path, cur_len, opt_path, min_len, is_chosen):
    if len(cur_path) == (len(points) - 1):
        # 只剩下终点
        cur_len += cal_similarity(points[-1], points[cur_path[-1]])
        if cur_len < min_len:
            min_len = cur_len
            for i in range(len(cur_path)):
                opt_path[i] = cur_path[i]
            opt_path[-1] = len(points) - 1
        return min_len

    point_num = len(points)
    for i in range(1, point_num - 1):
        if is_chosen[i]:
            continue
        tmp_len = cal_similarity(points[i], points[cur_path[-1]])
        cur_path.append(i)
        is_chosen[i] = True

        min_len = find_optimal_path(points, cur_path, cur_len + tmp_len, opt_path, min_len, is_chosen)

        is_chosen[i] = False
        cur_path.pop()

    return min_len
