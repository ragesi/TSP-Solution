# -*- coding: UTF-8 -*-
from qiskit import QuantumRegister, QuantumCircuit
import qiskit.circuit.library as lib
import numpy as np
import math as m
import cmath as cm

from qiskit.circuit.library import UnitaryGate

from utils import NOT_gate


def build_U_operator(qubit_num: int, dists: np.ndarray) -> UnitaryGate:
    """
    build the unitary operator that is used in the QPE algorithm
    :param qubit_num: integer, the number of qubits that are implemented by the unitary operator
    :param dists: the distance between the n-th node and any other node
    """
    matrix = np.eye(2 ** qubit_num, dtype=complex)
    for i in np.arange(len(dists)):
        tmp = i + 2 ** (qubit_num - 1)
        matrix[tmp][tmp] = cm.exp(1j * 2.0 * m.pi * dists[i])
    u_gate = lib.UnitaryGate(matrix)

    return u_gate


def QPE_U(control_num: int, target_num: int, dists: np.ndarray) -> QuantumCircuit:
    """
    the unitary operator in QPE, which is used to add the distance of every step
    :param control_num: integer, the number of control bits
    :param target_num: integer, the number of target bits
    :param dists: the n-th row in the distance adjacency whose size is 1 * n,
                  representing the distance between the n-th node and any other node
    """
    control = QuantumRegister(control_num)
    target = QuantumRegister(target_num)
    qc = QuantumCircuit(control, target)

    u_gate = build_U_operator(target_num + 1, dists)
    for i in np.arange(control_num):
        for _ in np.arange(2 ** (control_num - i - 1)):
            qc.append(u_gate, [control[i], *target])

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
        u_gate = build_U_operator(per_qram_num + 1, dist_adj[i])
        qc.append(NOT_gate.equal_to_int_NOT(i, per_qram_num, anc_num - 1), [*source, *anc[1:], anc[0]])
        qc.ccx(control[i], anc[0], anc[1])

        for j in np.arange(control_num):
            for _ in np.arange(2 ** (control_num - j - 1)):
                qc.append(u_gate, [anc[1], *target])

        qc.ccx(control[i], anc[0], anc[1])
        qc.append(NOT_gate.equal_to_int_NOT(i, per_qram_num, anc_num - 1), [*source, *anc[1:], anc[0]])

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


def inner_product():
    control = QuantumRegister(1)
    data = QuantumRegister(2)
    qc = QuantumCircuit(control, data)

    qc.h(control)
    qc.cswap(control[0], data[0], data[1])
    qc.h(control)

    return qc
