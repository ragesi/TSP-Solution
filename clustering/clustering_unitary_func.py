# -*- coding: UTF-8 -*-
from qiskit import QuantumRegister, QuantumCircuit
import math as m
import numpy as np

from utils import unitary_function


def cal_distance(point1, point2, precision):
    """
    calculating the distance of point1 and point2
    with preprocessing, point1_x >= point2_x and point1_y >= point2_y
    :param point1: list[point1_x, point1_y]
    :param point2: list[point2_x, point2_y]
    :param precision: integer, the number of bits occupied by distance
    :return: QuantumCircuit
    """
    control = QuantumRegister(precision)
    target = QuantumRegister(1)
    qc = QuantumCircuit(control, target)
    qc.h(control)
    qc.x(target)

    theta = round(1.0 * (point1[0] - point2[0] + point1[1] - point2[1]) * 60)
    theta = 2.0 * m.pi * theta / 64
    for i in np.arange(precision):
        for _ in np.arange(2 ** i):
            qc.cp(theta, control[i], target[0])

    unitary_function.QFT_dgr(qc, control, precision)

    return qc


def build_QRAM(dist_id, index_num, precision):
    source = QuantumRegister(precision)
    index = QuantumRegister(index_num)
    val = QuantumRegister(precision)
    qc = QuantumCircuit(source, index, val)

    # mapping
    for i in np.arange(index_num):
        if dist_id % 2 == 0:
            qc.x(index[i])
        dist_id = int(dist_id / 2)

    for i in np.arange(index_num - 1, -1, -1):
        if dist_id % 2 == 0:
            qc.x(index[i])
        dist_id = int(dist_id / 2)



def find_minimum_oracle(precision, dist_num):
    """
    the Oracle operator of finding minimum Grover algorithm
    :param precision: integer, the number of bits occupied by distance
    :param dist_num: integer, the number of distance waiting to be sorted
    :return: QuantumCircuit
    """
    info = QuantumRegister(precision)
    identity = QuantumRegister(m.ceil(dist_num))
    val = QuantumRegister(precision)

