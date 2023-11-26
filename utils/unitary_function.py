# -*- coding: UTF-8 -*-
from qiskit import QuantumRegister, QuantumCircuit
import qiskit.circuit.library as lib
import numpy as np
import math as m
import cmath as cm

from qiskit.circuit.library import UnitaryGate

from utils import NOT_gate


def QPE_U(control_num: int, target_num: int, dists: np.ndarray, anc_num: int) -> QuantumCircuit:
    """
    the unitary operator in QPE, which is used to add the distance of every step
    :param control_num: integer, the number of control bits
    :param target_num: integer, the number of target bits
    :param dists: the n-th row in the distance adjacency whose size is 1 * n,
                  representing the distance between the n-th node and any other node
    :param anc_num: integer, the number of ancilla bits
    """
    control = QuantumRegister(control_num)
    target = QuantumRegister(target_num)
    anc = QuantumRegister(anc_num)
    qc = QuantumCircuit(control, target, anc)

    for i in np.arange(len(dists)):
        qc.append(NOT_gate.equal_to_int_NOT(i, target_num, anc_num - 1), [*target, *anc[1:], anc[0]])

        for j in np.arange(control_num):
            for _ in np.arange(2 ** (control_num - j - 1)):
                qc.cp(2.0 * m.pi * dists[i], control[j], anc[0])

        qc.append(NOT_gate.equal_to_int_NOT(i, target_num, anc_num - 1).inverse(), [*target, *anc[1:], anc[0]])

    return qc


def custom_QPE_U(control_num: int, per_qram_num: int, anc_num: int, dist_adj: np.ndarray) -> QuantumCircuit:
    """
    custom unitary operator of QPE algorithm, according the route's source add distance dynamically
    :param control_num: integer, the precision of result
    :param per_qram_num: integer, the number of choice bit
    :param anc_num: integer, must be greater than 1
    :param dist_adj: distance adjacency
    """
    control = QuantumRegister(control_num)
    source = QuantumRegister(per_qram_num)
    target = QuantumRegister(per_qram_num)
    anc = QuantumRegister(anc_num)
    qc = QuantumCircuit(control, source, target, anc)

    for i in np.arange(len(dist_adj)):
        qc.append(NOT_gate.equal_to_int_NOT(i, per_qram_num, anc_num - 2), [*source, *anc[2:], anc[0]])

        for j in np.arange(len(dist_adj[i])):
            qc.append(NOT_gate.equal_to_int_NOT(j, per_qram_num, anc_num - 2), [*target, *anc[2:], anc[1]])

            for k in np.arange(control_num):
                qc.ccx(anc[0], control[k], anc[2])
                for _ in np.arange(2 ** (control_num - k - 1)):
                    qc.cp(2.0 * m.pi * dist_adj[i][j], anc[2], anc[1])
                qc.ccx(anc[0], control[k], anc[2])

            qc.append(NOT_gate.equal_to_int_NOT(j, per_qram_num, anc_num - 2).inverse(), [*target, *anc[2:], anc[1]])

        qc.append(NOT_gate.equal_to_int_NOT(i, per_qram_num, anc_num - 2).inverse(), [*source, *anc[2:], anc[0]])

    return qc


def grover_diffusion(qram_num, anc_num):
    qram = QuantumRegister(qram_num)
    anc = QuantumRegister(anc_num)
    res = QuantumRegister(1)
    qc = QuantumCircuit(qram, anc, res)

    qc.h(qram)
    qc.x(qram)
    qc.append(NOT_gate.custom_mcx(qram_num, anc_num), [*qram, *anc, *res])
    qc.x(qram)
    qc.h(qram)

    return qc


def inner_product():
    control = QuantumRegister(1)
    data = QuantumRegister(2)
    qc = QuantumCircuit(control, data)

    qc.h(control)
    qc.cswap(control[0], data[0], data[1])
    qc.h(control)

    return qc
