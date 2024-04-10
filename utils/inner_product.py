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


def cal_inner_product(vec_list_1, vec_list_2, env, backend):
    num_1 = len(vec_list_1)
    num_2 = len(vec_list_2)

    q = QuantumRegister(num_1 * num_2 * 3)
    cl = ClassicalRegister(num_1 * num_2)
    qc = QuantumCircuit(q, cl)

    for i in range(num_1):
        theta_1, phi_1 = to_bloch_state(vec_list_1[i])
        for j in range(num_2):
            theta_2, phi_2 = to_bloch_state(vec_list_2[j])

            qc.u(theta_1, phi_1, 0, q[i * (num_2 * 3) + j * 3 + 1])
            qc.u(theta_2, phi_2, 0, q[i * (num_2 * 3) + j * 3 + 2])

            qc.h(q[i * (num_2 * 3) + j * 3])
            qc.cswap(q[i * (num_2 * 3) + j * 3], q[i * (num_2 * 3) + j * 3 + 1], q[i * (num_2 * 3) + j * 3 + 2])
            qc.h(q[i * (num_2 * 3) + j * 3])

            qc.measure(q[i * (num_2 * 3) + j * 3], cl[i * num_2 + j])

    # base_theta, base_phi = to_bloch_state(vec_list_1)
    # for i in range(num_2):
    #     qc.u(base_theta, base_phi, 0, q[i * 3 + 1])
    #     cur_theta, cur_phi = to_bloch_state(vec_list_2[i])
    #     qc.u(cur_theta, cur_phi, 0, q[i * 3 + 2])
    #
    #     qc.h(q[i * 3])
    #     qc.cswap(q[i * 3], q[i * 3 + 1], q[i * 3 + 2])
    #     qc.h(q[i * 3])
    #
    # for i in range(num_2):
    #     qc.measure(q[i * 3], cl[i])

    job = execute.exec_qcircuit(qc, 20000, env, False, backend, False)
    return job
    # output = execute.get_output(job, env)
    #
    # output_dict = dict()
    # for item in output.items():
    #     output_dict[item[0][::-1]] = item[1]
    #
    # return output_dict


def get_max_inner_product(job, num_1, num_2, env):
    output = execute.get_output(job, env)

    values = [0 for _ in range(num_1 * num_2)]
    for item in output.items():
        tmp_key = item[0][::-1]
        for i in range(num_1 * num_2):
            values[i] += item[1] if tmp_key[i] == '0' else 0

    res = list()
    for i in range(num_1):
        res.append(values[i * num_2: (i + 1) * num_2].index(max(values[i * num_2: (i + 1) * num_2])))
    return res

    # res = [0 for _ in range(vec_num)]
    # for item in output.items():
    #     tmp_key = item[0][::-1]
    #     for i in range(vec_num):
    #         res[i] += item[1] if tmp_key[i] == '0' else 0
    # return res.index(max(res))


if __name__ == '__main__':
    base_vec = [3, 5]
    cur_vec_list = [[1, 6], [4, 3]]
    cal_inner_product(base_vec, cur_vec_list)
