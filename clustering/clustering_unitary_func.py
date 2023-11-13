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
    :param precision: integer, the precision of distance
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
