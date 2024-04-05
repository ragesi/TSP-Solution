from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from utils import execute

import math as m


def get_vec_range(vec_list):
    min_x = min([vec[0] for vec in vec_list])
    max_x = max([vec[0] for vec in vec_list])
    min_y = min([vec[1] for vec in vec_list])
    max_y = max([vec[1] for vec in vec_list])

    max_range = max(max_x - min_x, max_y - min_y)
    return max_range, min_x, min_y


def to_bloch_state(vec):
    theta = m.pi / 2 * (vec[0] + vec[1])
    phi = m.pi / 2 * (vec[0] - vec[1] + 1)
    return theta, phi


def cal_inner_product(base_vec, cur_vec_list, max_qubit_num=30):
    vec_num = len(cur_vec_list)
    if (vec_num * 3) > max_qubit_num:
        raise ValueError("Too many vectors!")

    q = QuantumRegister(vec_num * 3)
    cl = ClassicalRegister(vec_num)
    qc = QuantumCircuit(q, cl)

    base_theta, base_phi = to_bloch_state(base_vec)
    for i in range(vec_num):
        qc.u(base_theta, base_phi, 0, q[i * 3 + 1])
        cur_theta, cur_phi = to_bloch_state(cur_vec_list[i])
        qc.u(cur_theta, cur_phi, 0, q[i * 3 + 2])

        qc.h(q[i * 3])
        qc.cswap(q[i * 3], q[i * 3 + 1], q[i * 3 + 2])
        qc.h(q[i * 3])

    for i in range(vec_num):
        qc.measure(q[i * 3], cl[i])

    job = execute.exec_qcircuit(qc, 20000, 'sim', False, None, False)
    output = execute.get_output(job, 'sim')

    output_dict = dict()
    for item in output.items():
        output_dict[item[0][::-1]] = item[1]

    return output_dict


def get_max_inner_product(base_vec, cur_vec_list, max_qubit_num=30):
    output = cal_inner_product(base_vec, cur_vec_list, max_qubit_num)

    vec_num = len(cur_vec_list)
    res = [0 for _ in range(vec_num)]
    for item in output.items():
        for i in range(vec_num):
            res[i] += item[1] if item[0][i] == '0' else 0
    return res.index(max(res))


if __name__ == '__main__':
    base_vec = [3, 5]
    cur_vec_list = [[1, 6], [4, 3]]
    cal_inner_product(base_vec, cur_vec_list)
