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



