# -*- coding: UTF-8 -*-
import numpy as np
import math as m

from utils import NOT_gate


def QPE_U(qc, control, target, ancilla, dis_adj):
    """
    the U operator in QPE, which is used to add the distance of every step
    :param qc: QuantumCircuit
    :param control: QuantumRegister[int]
    :param target: QuantumRegister
    :param ancilla: QuantumRegister[int]
    :param dis_adj: adjacency matrix, [1 * 4]
    """
    # TODO: only 4 * 4 dis_adj is considered, it is better to expand

    # setting the first three element in dis_adj
    qc.cp(dis_adj[2] - dis_adj[0], control, target[0])
    qc.p(dis_adj[0], control)
    qc.cp(dis_adj[1] - dis_adj[0], control, target[1])

    # setting the last element in dis_adj
    delta_dis = dis_adj[3] - dis_adj[1] - dis_adj[2] + dis_adj[0]
    qc.ccx(control, target[0], ancilla)
    qc.cp(delta_dis, ancilla, target[1])
    qc.ccx(control, target[0], ancilla)


def QPE_U_dgr(qc, control, target, ancilla, dis_adj):
    """
    the inverse of QPE_U
    :param qc: QuantumCircuit
    :param control: QuantumRegister[int]
    :param target: QuantumRegister
    :param ancilla: QuantumRegister[int]
    :param dis_adj: adjacency matrix, [1 * 4]
    """
    # reverse the delta_dis
    delta_dis = -dis_adj[3] + dis_adj[1] + dis_adj[2] - dis_adj[0]
    qc.ccx(control, target[0], ancilla)
    qc.cp(delta_dis, ancilla, target[1])
    qc.ccx(control, target[0], ancilla)

    qc.cp(-dis_adj[1] + dis_adj[0], control, target[1])
    qc.p(-dis_adj[0], control)
    qc.cp(-dis_adj[2] + dis_adj[0], control, target[0])


def n_QPE_U(qc, control, source, target, ancilla, dis_adj):
    """
    the same as QPE_U but has multi-bits in control,
    which is used when the source of every step is unclear
    :param qc: QuantumCircuit
    :param control: QuantumRegister[int], one of the precision bit
    :param source: QuantumRegister, the source register for current step
    :param target: QuantumRegister
    :param ancilla: QuantumRegister, need 3 bits
    :param dis_adj: adjacency matrix, [3 * 4]
    """
    for i in np.arange(len(dis_adj)):
        NOT_gate.equal_to_int_NOT(qc, source, i, ancilla[0], ancilla[1:], len(source))
        qc.ccx(control, ancilla[0], ancilla[1])

        QPE_U(qc, ancilla[1], target, ancilla[2], dis_adj[i])

        qc.ccx(control, ancilla[0], ancilla[1])
        NOT_gate.equal_to_int_NOT(qc, source, i, ancilla[0], ancilla[1:], len(source))


def n_QPE_U_dgr(qc, control, source, target, ancilla, dis_adj):
    """
    the inverse of n_QPE_U
    :param qc: QuantumCircuit
    :param control: QuantumRegister[int]
    :param source: QuantumRegister
    :param target: QuantumRegister
    :param ancilla: QuantumRegister, need 3 bits
    :param dis_adj: adjacency matrix, [3 * 4]
    """

    for i in np.arange(len(dis_adj) - 1, -1, -1):
        NOT_gate.equal_to_int_NOT(qc, source, i, ancilla[0], ancilla[1:], len(source))
        qc.ccx(control, ancilla[0], ancilla[1])

        QPE_U_dgr(qc, ancilla[1], target, ancilla[2], dis_adj[i])

        qc.ccx(control, ancilla[0], ancilla[1])
        NOT_gate.equal_to_int_NOT(qc, source, i, ancilla[0], ancilla[1:], len(source))


def QFT(qc, target, target_num):
    """
    the operator of Quantum Fourier Transformation
    :param qc: QuantumCircuit
    :param target: QuantumRegister
    :param target_num: integer, the number of q bits
    """
    R_phis = [0]
    for i in np.arange(2, int(target_num + 1)):
        R_phis.append(2 / (2 ** i) * m.pi)
    for j in np.arange(int(target_num)):
        qc.h(target[int(j)])
        for k in np.arange(int(target_num - (j + 1))):
            qc.cp(R_phis[k + 1], target[int(j + k + 1)], target[int(j)])


def QFT_dgr(qc, target, target_num):
    """
    the inverse of QFT operator
    :param qc: QuantumCircuit
    :param target: QuantumRegister
    :param target_num: integer, the number of q bits
    """
    R_phis = [0]
    for i in np.arange(2, int(target_num + 1)):
        R_phis.append(-2 / (2 ** i) * m.pi)
    for j in np.arange(int(target_num)):
        for k in np.arange(int(j)):
            qc.cp(R_phis[int(j - k)], target[int(target_num - (k + 1))], target[int(target_num - (j + 1))])
        qc.h(target[int(target_num - (j + 1))])


def Grover_diffusion(qc, control, target, ancilla, control_num):
    """
    the amplitude amplification part of Grover
    :param qc: QuantumCircuit
    :param control: QuantumRegister
    :param target: QuantumRegister[int]
    :param ancilla: QuantumRegister[int], need (control_num - 1) bits
    :param control_num: integer, the number of control bits
    """
    for i in np.arange(control_num):
        qc.h(control[i])
    NOT_gate.zero_NOT(qc, control, target, ancilla, control_num)
    for i in np.arange(control_num - 1, -1, -1):
        qc.h(control[i])
