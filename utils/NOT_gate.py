# -*- coding: UTF-8 -*-
from qiskit import QuantumRegister, QuantumCircuit
import qiskit.circuit.library as lib

from typing import Optional
import numpy as np


# def n_NOT(qc, control, target, ancilla, control_num):
#     """
#     control-NOT gate, but the range of the number of control bits is from 1 to n
#     :param qc: QuantumCircuit
#     :param control: QuantumRegister
#     :param target: QuantumRegister[int]
#     :param ancilla: QuantumRegister, need (control_num - 2) bits
#     :param control_num: integer, the number of control bits
#     """
#     if control_num == 0:
#         qc.x(target)
#     elif control_num == 1:
#         qc.cx(control[0], target)
#     elif control_num == 2:
#         qc.ccx(control[0], control[1], target)
#     elif control_num > 2:
#         multi_NOT(qc, control, target, ancilla, control_num)


def zero_NOT(control_num: int) -> QuantumCircuit:
    """
    if the state of all control bits are True, the target bit is flipped
    :param control_num: integer, the number of control bits
    """
    control = QuantumRegister(control_num)
    target = QuantumRegister(1)
    qc = QuantumCircuit(control, target)

    qc.x(control)
    qc.mcx(control, target)
    qc.x(control)

    return qc


def custom_mcx(control_num: int, anc_num: int) -> QuantumCircuit:
    """
    custom mcx gate, dynamically determine the mode of mcx gate by the number of ancilla bits
    :param control_num: the number of control bits
    :param anc_num: the number of ancilla bits, use v-chain mode if ancilla is enough
    """
    control = QuantumRegister(control_num)
    anc = QuantumRegister(anc_num)
    res = QuantumRegister(1)
    qc = QuantumCircuit(control, anc, res)

    if anc_num >= (control_num - 2):
        qc.mcx(control, res[0], anc[:(control_num - 2)], mode='v-chain')
    else:
        qc.mcx(control, res[0])

    return qc


def equal_to_int_NOT(reference_state: int, control_num: int, anc_num: int) -> QuantumCircuit:
    """
    if the state of control register corresponds to the specified integer reference, the target bit is flipped
    :param reference_state: integer, the expected value of control bits
    :param control_num: integer, the number of control bits
    :param anc_num: integer, the number of ancilla bits
    """
    control = QuantumRegister(control_num)
    anc = QuantumRegister(anc_num)
    res = QuantumRegister(1)
    qc = QuantumCircuit(control, anc, res)

    x_list = []
    for i in np.arange(control_num):
        if reference_state % 2 == 0:
            x_list.append(control[i])
        reference_state = int(reference_state / 2)

    for i in np.arange(len(x_list)):
        qc.x(x_list[i])
    qc.append(custom_mcx(control_num, anc_num), [*control, *anc, *res])
    for i in np.arange(len(x_list) - 1, -1, -1):
        qc.x(x_list[i])

    return qc


def equal_NOT(control_num: int) -> QuantumCircuit:
    """
    if the values represented by two quantum registers are equal, flip the target bit
    :param control_num: integer, the number of qubits that to be compared
    """
    control1 = QuantumRegister(control_num)
    control2 = QuantumRegister(control_num)
    anc = QuantumRegister(control_num)
    target = QuantumRegister(1)
    qc = QuantumCircuit(control1, control2, anc, target)

    # if they are equal, the effect on ancilla is offset, so ancilla[i] will still be 0
    qc.cx(control1, anc)
    qc.cx(control2, anc)
    qc.append(zero_NOT(control_num), [*anc, *target])
    qc.cx(control2, anc)
    qc.cx(control1, anc)

    return qc

# def label_all_different(qc, control, threshold, ancilla, control_num):
#     """
#     using dynamic programming to label all bits with differences in the current bits or previous bits
#     :param qc: QuantumCircuit
#     :param control: QuantumRegister
#     :param threshold: binary string, whose length is control_num
#     :param ancilla: QuantumRegister, need (control_num + 2) bits
#     :param control_num: integer, the number of control
#     """
#     # only when all bits up to position i-th are equal, ancilla[i + 1] = 0
#     for i in np.arange(control_num):
#         # if control and threshold is different in i-th bit, setting i-th ancilla to 1
#         if threshold[i] == '1':
#             qc.x(ancilla[0])
#
#         if i == 0:
#             qc.cx(control[i], ancilla[i + 1])
#             qc.cx(ancilla[0], ancilla[i + 1])
#         else:
#             qc.cx(control[i], ancilla[-1])
#             qc.cx(ancilla[0], ancilla[-1])
#             OR_NOT(qc, [ancilla[i], ancilla[-1]], ancilla[i + 1], ancilla, 2)
#             qc.cx(ancilla[0], ancilla[-1])
#             qc.cx(control[i], ancilla[-1])
#
#         if threshold[i] == '1':
#             qc.x(ancilla[0])
#
#
# def label_all_different_dgr(qc, control, threshold, ancilla, control_num):
#     """
#     the inverse of label_all_different
#     :param qc: QuantumCircuit
#     :param control: QuantumRegister
#     :param threshold: binary string, whose length is control_num
#     :param ancilla: QuantumRegister, need (control_num + 2) bits
#     :param control_num: integer, the number of control
#     """
#     for i in np.arange(control_num - 1, -1, -1):
#         if threshold[i] == '1':
#             qc.x(ancilla[0])
#
#         if i == 0:
#             qc.cx(ancilla[0], ancilla[i + 1])
#             qc.cx(control[i], ancilla[i + 1])
#         else:
#             qc.cx(control[i], ancilla[-1])
#             qc.cx(ancilla[0], ancilla[-1])
#             OR_NOT_dgr(qc, [ancilla[i], ancilla[-1]], ancilla[i + 1], ancilla, 2)
#             qc.cx(ancilla[0], ancilla[-1])
#             qc.cx(control[i], ancilla[-1])
#
#         if threshold[i] == '1':
#             qc.x(ancilla[0])
#
#
# def less_than_int_NOT(qc, control, threshold, target, ancilla, control_num):
#     """
#     if control is less than the threshold, set the NOT gate on the target
#     :param qc: QuantumCircuit
#     :param control: QuantumRegister
#     :param threshold: binary string, whose length is control_num
#     :param target: QuantumRegister[int]
#     :param ancilla: QuantumRegister, need (control_num + 2) bits
#     :param control_num: integer, the number of control
#     """
#     # using dynamic programming to label all bits
#     label_all_different(qc, control, threshold, ancilla, control_num)
#
#     # remarking all bits, leaving only the mark for the first occurrence of difference
#     for i in np.arange(control_num, 1, -1):
#         qc.cx(ancilla[i - 1], ancilla[i])
#
#     # label the first occurrence that control less than threshold
#     for i in np.arange(control_num):
#         qc.x(control[i])
#         if threshold[i] == '1':
#             qc.x(ancilla[0])
#
#         n_NOT(qc, [control[i], ancilla[0], ancilla[i + 1]], target, ancilla[-1:], 3)
#
#         if threshold[i] == '1':
#             qc.x(ancilla[0])
#         qc.x(control[i])
#
#     # dgr
#     for i in np.arange(2, control_num + 1):
#         qc.cx(ancilla[i - 1], ancilla[i])
#
#     label_all_different_dgr(qc, control, threshold, ancilla, control_num)
#
#
# def less_than_int_NOT_dgr(qc, control, threshold, target, ancilla, control_num):
#     """
#     the inverse of less_than_int_NOT
#     :param qc: QuantumCircuit
#     :param control: QuantumRegister
#     :param threshold: binary string, whose length is control_num
#     :param target: QuantumRegister[int]
#     :param ancilla: QuantumRegister, need (control_num + 2) bits
#     :param control_num: integer, the number of control
#     """
#     label_all_different_dgr(qc, control, threshold, ancilla, control_num)
#
#     for i in np.arange(2, control_num + 1):
#         qc.cx(ancilla[i - 1], ancilla[i])
#
#     for i in np.arange(control_num - 1, -1, -1):
#         qc.x(control[i])
#         if threshold[i] == '1':
#             qc.x(ancilla[0])
#
#         n_NOT(qc, [control[i], ancilla[0], ancilla[i + 1]], target, ancilla[-1:], 3)
#
#         if threshold[i] == '1':
#             qc.x(ancilla[0])
#         qc.x(control[i])
#
#     for i in np.arange(control_num, 1, -1):
#         qc.cx(ancilla[i - 1], ancilla[i])
#
#     label_all_different(qc, control, threshold, ancilla, control_num)
