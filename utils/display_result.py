# -*- coding: UTF-8 -*-
from qiskit import Aer, execute
import numpy as np

S_simulator = Aer.backends(name='statevector_simulator')[0]
M_simulator = Aer.backends(name='qasm_simulator')[0]


def Measurement(quantumcircuit, **kwargs):
    """
    Executes a measurement(s) of a QuantumCircuit object for tidier printing
    :param quantumcircuit: QuantumCircuit
    :param kwargs: shots: integer, number of trials to execute for the measurement(s)
                   return_M: Bool, indicates whether to return the Dictionary object containing measurement result
                   print_M: Bool, indicates whether to print the measurement results
                   column: Bool, prints each state in a vertical column
    """
    p_M = True
    S = 1
    ret = False
    NL = False
    if 'shots' in kwargs:
        S = int(kwargs['shots'])
    if 'return_M' in kwargs:
        ret = kwargs['return_M']
    if 'print_M' in kwargs:
        p_M = kwargs['print_M']
    if 'column' in kwargs:
        NL = kwargs['column']
    M1 = execute(quantumcircuit, M_simulator, shots=S).result().get_counts(quantumcircuit)
    M2 = {}
    k1 = list(M1.keys())
    v1 = list(M1.values())
    for k in np.arange(len(k1)):
        key_list = list(k1[k])
        new_key = ''
        for j in np.arange(len(key_list)):
            new_key = new_key + key_list[len(key_list) - (j + 1)]
        M2[new_key] = v1[k]
    if p_M:
        k2 = list(M2.keys())
        v2 = list(M2.values())
        measurements = ''
        for i in np.arange(len(k2)):
            m_str = str(v2[i]) + '|'
            for j in np.arange(len(k2[i])):
                if k2[i][j] == '0':
                    m_str = m_str + '0'
                if k2[i][j] == '1':
                    m_str = m_str + '1'
                if k2[i][j] == ' ':
                    m_str = m_str + '>|'
            m_str = m_str + '>   '
            if NL:
                m_str = m_str + '\n'
            measurements = measurements + m_str
        print(measurements)
        return measurements
    if ret:
        return M2
