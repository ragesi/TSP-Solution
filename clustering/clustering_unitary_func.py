# -*- coding: UTF-8 -*-
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
import qiskit.circuit.library as lib
import math as m
import numpy as np

from utils import display_result as disp


def cal_range(points):
    tmp_x = sorted([point[0] for point in points])
    tmp_y = sorted([point[1] for point in points])
    return [tmp_x[0], tmp_x[-1]], [tmp_y[0], tmp_y[-1]]


def equal_to_int_NOT(real, qubit_num):
    control = QuantumRegister(qubit_num)
    res = QuantumRegister(1)
    qc = QuantumCircuit(control, res)

    x_list = []
    for i in np.arange(qubit_num):
        if real % 2 == 0:
            x_list.append(control[i])
        real = int(real / 2)

    for i in np.arange(len(x_list)):
        qc.x(x_list[i])
    qc.mcx(control, res)
    for i in np.arange(len(x_list) - 1, -1, -1):
        qc.x(x_list[i])

    return qc


def to_phase_state(source, targets):
    x_range, y_range = cal_range([source, *targets])
    x_width = x_range[1] - x_range[0]
    y_width = y_range[1] - y_range[0]

    source = [0.5 * (source[0] - x_range[0]) / x_width, 0.5 * (source[1] - y_range[0]) / y_width]
    for i in np.arange(len(targets)):
        targets[i] = [0.5 * (targets[i][0] - x_range[0]) / x_width, 0.5 * (targets[i][1] - y_range[0]) / y_width]

    return source, targets


def int_to_qubit(real, qubit_num):
    qc = QuantumCircuit(qubit_num)

    for i in np.arange(qubit_num):
        if real % 2 == 1:
            qc.x(i)
        real = int(real / 2)

    return qc


def compare_int(comp1, comp2, precision):
    val = QuantumRegister(precision)
    anc = QuantumRegister(precision - 1)
    res = QuantumRegister(1)
    qc = QuantumCircuit(val, anc, res)

    comp1 = round(comp1 * (2 ** precision))
    comp2 = round(comp2 * (2 ** precision))
    qc.append(int_to_qubit(comp1, precision), val)
    qc.append(lib.IntegerComparator(precision, comp2), [*val, *res, *anc])

    return qc


def QPE_U(precision, theta):
    control = QuantumRegister(precision)
    target = QuantumRegister(1)
    qc = QuantumCircuit(control, target)

    theta = round(theta * (2 ** precision)) * 2.0 * m.pi / (2 ** precision)
    for i in np.arange(precision):
        for _ in np.arange(2 ** (precision - i - 1)):
            qc.cp(theta, control[i], target[0])

    return qc


def cal_dist(source, target, precision):
    id_anc = QuantumRegister(1)
    dist = QuantumRegister(precision)
    dist_abs = QuantumRegister(precision)
    dist_anc = QuantumRegister(1)
    qpe_anc = QuantumRegister(1)
    anc = QuantumRegister(precision - 1)
    qc = QuantumCircuit(id_anc, dist, dist_abs, qpe_anc, anc)

    for i in np.arange(len(source)):
        qc.append(compare_int(source[i], target[i], precision), [*dist_abs, *anc, *dist_anc])

        for j in np.arange(precision):
            qc.ccx(dist[j], dist_anc[0], qpe_anc[0])
            theta = round((source[i] - target[i]) * (2 ** precision)) * 2.0 * m.pi / (2 ** precision)
            for _ in np.arange(2 ** (precision - j - 1)):
                qc.cp(theta, qpe_anc[0], id_anc[0])
            qc.ccx(dist[j], dist_anc[0], qpe_anc[0])

            qc.x(dist_anc[0])
            qc.ccx(dist[j], dist_anc[0], qpe_anc[0])
            theta = round((target[i] - source[i]) * (2 ** precision)) * 2.0 * m.pi / (2 ** precision)
            for _ in np.arange(2 ** (precision - j - 1)):
                qc.cp(theta, qpe_anc[0], id_anc[0])
            qc.ccx(dist[j], dist_anc[0], qpe_anc[0])
            qc.x(dist_anc[0])

        qc.append(compare_int(source[i], target[i], precision).inverse(), [*dist_abs, *anc, *dist_anc])

    return qc


def build_QRAM(dist_id, index_num, theta, precision):
    index = QuantumRegister(index_num)
    id_anc = QuantumRegister(1)
    dist = QuantumRegister(precision)
    qc = QuantumCircuit(index, id_anc, dist)

    qc.append(equal_to_int_NOT(dist_id, index_num), [*index, *id_anc])
    qc.mcx(index, id_anc)

    for i in np.arange(precision):
        for _ in np.arange(2 ** (precision - i - 1)):
            qc.cp(theta, dist[i], id_anc[0])

    qc.mcx(index, id_anc)
    qc.append(equal_to_int_NOT(dist_id, index_num).inverse(), [*index, *id_anc])

    qc.append(lib.QFT(precision, inverse=True, do_swaps=False), dist)

    return qc


def find_min_oracle(source, targets, threshold, precision, index_num):
    # TODO: 此处的距离计算过程交由经典计算机执行，可以尝试使用交换门计算距离，但需要验证交换门计算出的距离是否准确
    # TODO: 若交换门计算的距离的准确性可以接受，那么就使用交换门计算；若不可以接受，则因为网上现在并没有使用经典计算机计算距离的先例，所以就只能使用未完成的方式计算了
    dist_num = len(targets)
    # id_reg_num = m.ceil(m.log2(dist_num))
    index = QuantumRegister(index_num)
    id_anc = QuantumRegister(1)
    dist = QuantumRegister(precision)
    anc = QuantumRegister(precision - 1)
    res = QuantumRegister(1)
    qc = QuantumCircuit(index, id_anc, dist, anc, res)

    # start oracle
    for i in np.arange(dist_num):
        theta = round((abs(source[0] - targets[i][0]) + abs(source[1] - targets[i][1])) * (2 ** precision))
        theta = 2.0 * m.pi * theta / (2 ** precision)
        qc.append(build_QRAM(i, index_num, theta, precision), [*index, *id_anc, *dist])

        qc.append(lib.IntegerComparator(precision, threshold, False), [*dist, *res, *anc])

        qc.append(build_QRAM(i, index_num, theta, precision).inverse(), [*index, *id_anc, *dist])


def find_min_diffusion(index_num):
    index = QuantumRegister(index_num)
    res = QuantumRegister(1)
    qc = QuantumCircuit(index, res)

    qc.h(index)
    qc.x(index)
    qc.mcx(index, res)
    qc.x(index)
    qc.h(index)
    return qc


def find_min_grover(source, targets, threshold, precision, iter_num):
    dist_num = len(targets)
    index_num = m.ceil(m.log2(dist_num)) + 1
    index = QuantumRegister(index_num)
    id_anc = QuantumRegister(1)
    dist = QuantumRegister(precision)
    anc = QuantumRegister(precision - 1)
    res = QuantumRegister(1)
    qc = QuantumCircuit(index, id_anc, dist, anc, res)
    # initialization
    qc.h(index)
    qc.x(res)
    qc.h(res)
    # map points' x and y to the range of 0 to 0.5
    source, targets = to_phase_state(source, targets)

    for _ in np.arange(iter_num):
        qc.append(find_min_oracle(source, targets, threshold, precision, index_num),
                  [*index, *id_anc, *dist, *anc, *res])
        qc.append(find_min_diffusion(index_num), [*index, *res])

    output = disp.Measurement(qc, return_M=False, print_M=True, shots=1)
    return output[0]


def find_minimum(source, targets, precision):
    # map points' x and y to the range of 0 to 0.5
    source, targets = to_phase_state(source, targets)
    threshold = round((abs(source[0] - targets[0][0]) + abs(source[1] - targets[0][1])) * (2 ** precision))
    find_min_grover(source, targets, threshold, precision, 1)


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
