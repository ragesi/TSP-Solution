# -*- coding: UTF-8 -*-
import numpy as np


def multi_NOT(qc, control, target, anc, control_num):
    """
    Multi-control-not gate, doing the same as ccx() but accept more than 2 control bits
    :param qc: QuantumCircuit
    :param control: QuantumRegister
    :param target: QuantumRegister[integer], which is an element but not a list
    :param anc: QuantumRegister, need len(control) - 2 bits
    :param control_num: integer, the number of control bits
    """
    instructions = []
    active_ancilla = []
    q_unused = []
    q = 0
    a = 0
    while (control_num > 0) or (len(q_unused) != 0) or (len(active_ancilla) != 0):
        if control_num > 0:
            if (control_num - 2) >= 0:
                instructions.append([control[q], control[q + 1], anc[a]])
                active_ancilla.append(a)
                a += 1
                q += 2
                control_num = control_num - 2
            if (control_num - 2) == -1:
                q_unused.append(q)
                control_num = control_num - 1
        elif len(q_unused) != 0:
            if len(active_ancilla) != 1:
                instructions.append([control[q], anc[active_ancilla[0]], anc[a]])
                del active_ancilla[0]
                del q_unused[0]
                active_ancilla.append(a)
                a = a + 1
            else:
                instructions.append([control[q], anc[active_ancilla[0]], target])
                del active_ancilla[0]
                del q_unused[0]
        elif len(active_ancilla) != 0:
            if len(active_ancilla) > 2:
                instructions.append([anc[active_ancilla[0]], anc[active_ancilla[1]], anc[a]])
                active_ancilla.append(a)
                del active_ancilla[0]
                del active_ancilla[0]
                a = a + 1
            elif len(active_ancilla) == 2:
                instructions.append([anc[active_ancilla[0]], anc[active_ancilla[1]], target])
                del active_ancilla[0]
                del active_ancilla[0]
    for i in np.arange(len(instructions)):
        qc.ccx(instructions[i][0], instructions[i][1], instructions[i][2])
    del instructions[-1]
    for i in np.arange(len(instructions)):
        qc.ccx(instructions[0 - (i + 1)][0], instructions[0 - (i + 1)][1], instructions[0 - (i + 1)][2])


def n_NOT(qc, control, target, ancilla, control_num):
    """
    control-NOT gate, but the range of the number of control bits is from 1 to n
    :param qc: QuantumCircuit
    :param control: QuantumRegister
    :param target: QuantumRegister[int]
    :param ancilla: QuantumRegister, need (control_num - 2) bits
    :param control_num: integer, the number of control bits
    """
    if control_num == 0:
        qc.x(target)
    elif control_num == 1:
        qc.cx(control[0], target)
    elif control_num == 2:
        qc.ccx(control[0], control[1], target)
    elif control_num > 2:
        multi_NOT(qc, control, target, ancilla, control_num)


def zero_NOT(qc, control, target, ancilla, control_num):
    """
    if control bits are all 0, set the NOT gate on the target
    :param qc: QuantumCircuit
    :param control: QuantumRegister
    :param target: QuantumRegister[int]
    :param ancilla: QuantumRegister, need (control_num - 2) bits
    :param control_num: integer, the number of control bits
    """
    for i in np.arange(control_num):
        qc.x(control[i])

    n_NOT(qc, control, target, ancilla, control_num)

    for i in np.arange(control_num):
        qc.x(control[i])


def OR_NOT(qc, control, target, ancilla, control_num):
    """
    if there is at least one 1 in control register, setting the NOT gat on the target
    :param qc: QuantumCircuit
    :param control: QuantumRegister
    :param target: QuantumRegister[int]
    :param ancilla: QuantumRegister, need (control_num - 2) bits
    :param control_num: integer, the number of control bits
    """
    zero_NOT(qc, control, target, ancilla, control_num)
    qc.x(target)


def OR_NOT_dgr(qc, control, target, ancilla, control_num):
    """
    the inverse of OR_NOT gate
    :param qc: QuantumCircuit
    :param control: QuantumRegister
    :param target: QuantumRegister[int]
    :param ancilla: QuantumRegister, need (control_num - 2) bits
    :param control_num: integer, the number of control bits
    """
    qc.x(target)
    zero_NOT(qc, control, target, ancilla, control_num)


def equal_to_int_NOT(qc, control, real, target, ancilla, control_num):
    """
    Determine whether the binary control and integer are equal. If they are equal, set the NOT gate on the target
    :param qc: QuantumCircuit
    :param control: QuantumRegister
    :param real: integer
    :param target: QuantumRegister[int]
    :param ancilla: QuantumRegister, need (control_num - 2) bits
    :param control_num: integer, the number of control bits
    """
    processing_list = []
    for i in np.arange(control_num - 1, 0, -1):
        if int(real / (2 ** i)) == 0:
            processing_list.append(control[control_num - i - 1])
        real %= (2 ** i)
    if real % 2 == 0:
        processing_list.append(control[control_num - 1])

    for i in np.arange(len(processing_list)):
        qc.x(processing_list[i])

    n_NOT(qc, control, target, ancilla, control_num)

    for i in np.arange(len(processing_list) - 1, -1, -1):
        qc.x(processing_list[i])


def equal_NOT(qc, control1, control2, target, ancilla, control_num):
    """
    if control1 is same as control2, set the NOT gate on the target
    :param qc: QuantumCircuit
    :param control1: QuantumRegister, the left side of equal condition
    :param control2: QuantumRegister, the right side of equal condition
    :param target: QuantumRegister[int]
    :param ancilla: QuantumRegister, need (2 * (control_num - 1)) bits
    :param control_num: integer, the number of control1 bits, also is the number of control2 bits
    """
    for i in np.arange(control_num):
        # if they are equal, the effect on ancilla is offset, so ancilla[i] will be 0
        qc.cx(control1[i], ancilla[i])
        qc.cx(control2[i], ancilla[i])

    zero_NOT(qc, ancilla[0: control_num], target, ancilla[control_num:], control_num)

    for i in np.arange(control_num - 1, -1, -1):
        qc.cx(control2[i], ancilla[i])
        qc.cx(control1[i], ancilla[i])


def label_all_different(qc, control, threshold, ancilla, control_num):
    """
    using dynamic programming to label all bits with differences in the current bits or previous bits
    :param qc: QuantumCircuit
    :param control: QuantumRegister
    :param threshold: binary string, whose length is control_num
    :param ancilla: QuantumRegister, need (control_num + 2) bits
    :param control_num: integer, the number of control
    """
    # only when all bits up to position i-th are equal, ancilla[i + 1] = 0
    for i in np.arange(control_num):
        # if control and threshold is different in i-th bit, setting i-th ancilla to 1
        if threshold[i] == '1':
            qc.x(ancilla[0])

        if i == 0:
            qc.cx(control[i], ancilla[i + 1])
            qc.cx(ancilla[0], ancilla[i + 1])
        else:
            qc.cx(control[i], ancilla[-1])
            qc.cx(ancilla[0], ancilla[-1])
            OR_NOT(qc, [ancilla[i], ancilla[-1]], ancilla[i + 1], ancilla, 2)
            qc.cx(ancilla[0], ancilla[-1])
            qc.cx(control[i], ancilla[-1])

        if threshold[i] == '1':
            qc.x(ancilla[0])


def label_all_different_dgr(qc, control, threshold, ancilla, control_num):
    """
    the inverse of label_all_different
    :param qc: QuantumCircuit
    :param control: QuantumRegister
    :param threshold: binary string, whose length is control_num
    :param ancilla: QuantumRegister, need (control_num + 2) bits
    :param control_num: integer, the number of control
    """
    for i in np.arange(control_num - 1, -1, -1):
        if threshold[i] == '1':
            qc.x(ancilla[0])

        if i == 0:
            qc.cx(ancilla[0], ancilla[i + 1])
            qc.cx(control[i], ancilla[i + 1])
        else:
            qc.cx(control[i], ancilla[-1])
            qc.cx(ancilla[0], ancilla[-1])
            OR_NOT_dgr(qc, [ancilla[i], ancilla[-1]], ancilla[i + 1], ancilla, 2)
            qc.cx(ancilla[0], ancilla[-1])
            qc.cx(control[i], ancilla[-1])

        if threshold[i] == '1':
            qc.x(ancilla[0])


def less_than_int_NOT(qc, control, threshold, target, ancilla, control_num):
    """
    if control is less than the threshold, set the NOT gate on the target
    :param qc: QuantumCircuit
    :param control: QuantumRegister
    :param threshold: binary string, whose length is control_num
    :param target: QuantumRegister[int]
    :param ancilla: QuantumRegister, need (control_num + 2) bits
    :param control_num: integer, the number of control
    """
    # using dynamic programming to label all bits
    label_all_different(qc, control, threshold, ancilla, control_num)

    # remarking all bits, leaving only the mark for the first occurrence of difference
    for i in np.arange(control_num, 1, -1):
        qc.cx(ancilla[i - 1], ancilla[i])

    # label the first occurrence that control less than threshold
    for i in np.arange(control_num):
        qc.x(control[i])
        if threshold[i] == '1':
            qc.x(ancilla[0])

        n_NOT(qc, [control[i], ancilla[0], ancilla[i + 1]], target, ancilla[-1:], 3)

        if threshold[i] == '1':
            qc.x(ancilla[0])
        qc.x(control[i])

    # dgr
    for i in np.arange(2, control_num + 1):
        qc.cx(ancilla[i - 1], ancilla[i])

    label_all_different_dgr(qc, control, threshold, ancilla, control_num)


def less_than_int_NOT_dgr(qc, control, threshold, target, ancilla, control_num):
    """
    the inverse of less_than_int_NOT
    :param qc: QuantumCircuit
    :param control: QuantumRegister
    :param threshold: binary string, whose length is control_num
    :param target: QuantumRegister[int]
    :param ancilla: QuantumRegister, need (control_num + 2) bits
    :param control_num: integer, the number of control
    """
    label_all_different_dgr(qc, control, threshold, ancilla, control_num)

    for i in np.arange(2, control_num + 1):
        qc.cx(ancilla[i - 1], ancilla[i])

    for i in np.arange(control_num - 1, -1, -1):
        qc.x(control[i])
        if threshold[i] == '1':
            qc.x(ancilla[0])

        n_NOT(qc, [control[i], ancilla[0], ancilla[i + 1]], target, ancilla[-1:], 3)

        if threshold[i] == '1':
            qc.x(ancilla[0])
        qc.x(control[i])

    for i in np.arange(control_num, 1, -1):
        qc.cx(ancilla[i - 1], ancilla[i])

    label_all_different(qc, control, threshold, ancilla, control_num)
