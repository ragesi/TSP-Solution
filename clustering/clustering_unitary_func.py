# -*- coding: UTF-8 -*-
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
import qiskit.circuit.library as lib
import math as m
import numpy as np

from utils import unitary_function as uf, util, display_result as disp


def cal_range(points):
    tmp_x = sorted([point[0] for point in points])
    tmp_y = sorted([point[1] for point in points])
    return [tmp_x[0], tmp_x[-1]], [tmp_y[0], tmp_y[-1]]


def build_U_operator(qubit_num, dist_num, source, targets):
    matrix_size = 2 ** qubit_num
    np.identity(matrix_size, float)
    x_range, y_range =
    for i in np.arange(dist_num):


def to_phase_state(source, targets):
    x_range, y_range = cal_range([source, *targets])
    x_width = x_range[1] - x_range[0]
    y_width = y_range[1] - y_range[0]

    source = [0.5 * (source[0] - x_range[0]) / x_width, 0.5 * (source[1] - y_range[0]) / y_width]
    for i in np.arange(len(targets)):
        targets[i] = [0.5 * (targets[i][0] - x_range[0]) / x_width, 0.5 * (targets[i][1] - y_range[0]) / y_width]

    return source, targets


def cal_dist(source, target, precision):



def find_min_oracle(source, targets, threshold, precision):
    dist_num = len(targets)
    id_reg_num = m.ceil(m.log2(dist_num))
    index = QuantumRegister(id_reg_num)
    id_anc = QuantumRegister(1)
    dist = QuantumRegister(precision)
    dist_abs = QuantumRegister(precision)
    qpe_anc = QuantumRegister(1)
    dist_comp = QuantumRegister(1)
    anc = QuantumRegister(min(id_reg_num - 2, precision - 1))
    res = QuantumRegister(1)
    qc = QuantumCircuit(index, id_anc, dist, dist_abs, qpe_anc, dist_comp, anc, res)
    # initialization
    qc.h(index)
    qc.x(res)
    qc.h(res)
    # map points' x and y to the range of 0 to 0.5
    source, targets = to_phase_state(source, targets)

    # start oracle
    for i in np.arange(dist_num):
        qc.mcx(index, id_anc, anc[:(id_reg_num - 2)], mode='v-chain')

        # calculate distance between source and current target
        # calculate the absolute value for delta x and delta y respectively


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
        for _ in np.arange(2 ** (precision - i - 1)):
            qc.cp(theta, control[i], target[0])

    qc.append(lib.QFT(num_qubits=precision, do_swaps=False, inverse=True), control)

    return qc


def build_QRAM(dist_id, index_num, precision):
    source = QuantumRegister(precision)
    index = QuantumRegister(index_num)
    val = QuantumRegister(precision)
    ancilla = QuantumRegister(index_num - 1)
    qc = QuantumCircuit(source, index, val)

    # mapping
    for i in np.arange(index_num):
        if dist_id % 2 == 0:
            qc.x(index[i])
        dist_id = int(dist_id / 2)

    for i in np.arange(precision):
        qc.mcx([source[i], *index], val[i], ancilla, mode='v-chain')

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
