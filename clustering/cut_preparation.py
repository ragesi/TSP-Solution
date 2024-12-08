import math as m
import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import laplacian
from sklearn.neighbors import NearestNeighbors
import pandas as pd

from qiskit import QuantumRegister, QuantumCircuit
import qiskit.circuit.library as lib


def build_gaussian_adj(adj_matrix: np.ndarray, point_num: int) -> np.ndarray:
    """
    Transform all elements in adjacency matrix into the form of gaussian
    :param adj_matrix: numpy array
    :param point_num: int
    :return: adj_matrix in the form of gaussian
    """
    min_element = adj_matrix.min()
    max_element = adj_matrix.max()
    sigma = (max_element - min_element) * 0.15

    adj_transform = np.zeros((point_num, point_num))
    for i in range(point_num):
        for j in range(point_num):
            if adj_matrix[i][j] != 0.0:
                adj_transform[i][j] = np.exp(-np.square(np.array(adj_matrix[i][j]) / sigma) / 2)
    return adj_transform


def build_adj_matrix(points: list, point_num: int) -> np.ndarray:
    """
    Build adjacency matrix from points
    :param points: list
    :param point_num: int
    :return: numpy array
    """
    adj_matrix = np.zeros((point_num, point_num))
    for i in range(point_num):
        for j in range(i + 1, point_num):
            adj_matrix[i][j] = adj_matrix[j][i] = np.linalg.norm(np.array(points[i]) - np.array(points[j]))
    # transfer to gaussian
    adj_matrix = build_gaussian_adj(adj_matrix, point_num)
    # enlarge all elements from the range of [0, 1] to [0, 10]
    adj_matrix *= 10
    return adj_matrix


def build_deg_matrix(adj_matrix: np.ndarray, point_num: int) -> list:
    """
    Build degree matrix from adjacency matrix
    :param point_num: int
    :param adj_matrix: numpy array
    :return: list
    """
    deg_matrix = list()
    for i in range(point_num):
        deg_matrix.append(sum(adj_matrix[i]))

    # normalize to the range of [0, 1]
    min_deg = min(deg_matrix)
    max_deg = max(deg_matrix)
    delta_deg = max_deg - min_deg
    for i in range(point_num):
        deg_matrix[i] = (deg_matrix[i] - min_deg) / delta_deg

    return deg_matrix


def scaling_up_deg_matrix(deg_matrix: list, point_num: int) -> list:
    """
    Amplify degree to distinguish each of them
    :param point_num: int
    :param deg_matrix: list
    """
    large_deg_matrix = list()
    for i in range(point_num):
        large_deg_matrix.append(deg_matrix[i] * 10 + 1)
    return large_deg_matrix


# def scaling_down_deg_matrix(deg_matrix, precision, point_num):
#     """
#     Normalize degree to the multiple of 2
#     :param precision: int, the precision of the QPE algorithm
#     :param deg_matrix: numpy array
#     :param point_num: int
#     :return: list
#     """
#     norm_deg_matrix = list()
#     sum_deg = sum(deg_matrix)
#     for i in range(point_num):
#         tmp_ele = max(round(deg_matrix[i] / sum_deg * (2 ** (precision - 1))), 1)
#         tmp_ele /= (2 ** precision)
#         norm_deg_matrix.append(tmp_ele)
#     return norm_deg_matrix


def scaling_down_deg_matrix(deg_matrix: list, norm_threshold: float, point_num: int) -> list:
    """
    normalizing degree
    :param deg_matrix: list
    :param norm_threshold: float
    :param point_num: int
    """
    small_deg_matrix = list()
    sum_deg = sum(deg_matrix)
    for i in range(point_num):
        small_deg_matrix.append(deg_matrix[i] / sum_deg * norm_threshold)
    return small_deg_matrix


def qpe(precision: int, small_deg_matrix: list, point_num: int):
    """
    Use the QPE algorithm to determine whether the sum of degree in cluster 0 is larger than the sum of degree in cluster 1
    :param precision:
    :param small_deg_matrix:
    :param point_num:
    :return:
    """
    qram = QuantumRegister(point_num)
    eigen_vec = QuantumRegister(1)
    eigen_val = QuantumRegister(precision)
    anc = QuantumRegister(1)
    qc = QuantumCircuit(qram, eigen_vec, eigen_val, anc)

    qc.h(eigen_val)
    qc.x(eigen_vec)
    for i in range(point_num):
        for j in range(precision - 1, -1, -1):
            qc.ccx(qram[i], eigen_val[j], anc[0])

            for _ in range(2 ** (precision - j - 1)):
                qc.cp(2 * m.pi * small_deg_matrix[i], anc[0], eigen_vec[0])

            qc.ccx(qram[i], eigen_val[j], anc[0])

            # If qram[i] == 0
            qc.x(qram[i])
            qc.ccx(qram[i], eigen_val[j], anc[0])
            for _ in range(2 ** (precision - j - 1)):
                qc.cp(-2 * m.pi * small_deg_matrix[i], anc[0], eigen_vec[0])
            qc.ccx(qram[i], eigen_val[j], anc[0])
            qc.x(qram[i])

    for j in range(precision - 1, -1, -1):
        for _ in range(2 ** (precision - j - 1)):
            qc.p(2 * m.pi * 0.25, eigen_val[j])

    qc.append(lib.QFT(precision, do_swaps=False, inverse=True), eigen_val)

    return qc
