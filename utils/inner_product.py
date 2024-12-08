from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from utils import execute, util

import math as m


# def normalization(self, point) -> list:
#     """
#     executing the normalization of points
#     :param point: list
#     """
#     return [(point[0] - self.x_range[0]) / self.range, (point[1] - self.y_range[0]) / self.range]


def normalization(point, x_min, y_min, range) -> list:
    """
    executing the normalization of points
    :param point: list
    :param x_min: the minimum x value
    :param y_min: the minimum y value
    :param range: the range of all points
    """
    return [(point[0] - x_min) / range, (point[1] - y_min) / range]


def to_bloch_state(vec):
    theta = m.pi / 2 * (vec[0] + vec[1])
    phi = m.pi / 2 * (vec[0] - vec[1] + 1)
    return theta, phi


def cal_inner_product(vec_list_1, vec_list_2, task_num_per_circuit, env, backend, print_detail=False):
    q = QuantumRegister(task_num_per_circuit * 3)
    cl = ClassicalRegister(task_num_per_circuit)
    qc = QuantumCircuit(q, cl)

    vec_num = len(vec_list_1)
    bloch_state_1 = [to_bloch_state(vec) for vec in vec_list_1]
    bloch_state_2 = [to_bloch_state(vec) for vec in vec_list_2]
    task_idx = 0
    for i in range(vec_num):
        qc.u(bloch_state_1[i][0], bloch_state_1[i][1], 0, q[task_idx * 3 + 1])
        qc.u(bloch_state_2[i][0], bloch_state_2[i][1], 0, q[task_idx * 3 + 2])

        qc.h(q[task_idx * 3])
        qc.cswap(q[task_idx * 3], q[task_idx * 3 + 1], q[task_idx * 3 + 2])
        qc.h(q[task_idx * 3])

        qc.measure(q[task_idx * 3], cl[task_idx])

        task_idx += 1

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

    job = execute.exec_qcircuit(qc, 20000, env, False, backend, print_detail)
    return job
    # output = execute.get_output(job, env)
    #
    # output_dict = dict()
    # for item in output.items():
    #     output_dict[item[0][::-1]] = item[1]
    #
    # return output_dict


def get_inner_product_result(job, task_num_per_circuit, env):
    output = execute.get_output(job, env)

    values = [0 for _ in range(task_num_per_circuit)]
    for item in output.items():
        tmp_key = item[0]
        if env != 'sim':
            tmp_key = util.int_to_binary(item[0], task_num_per_circuit)
        tmp_key = tmp_key[::-1]
        for i in range(task_num_per_circuit):
            values[i] += item[1] if tmp_key[i] == '0' else 0

    return values

    # res = list()
    # for i in range(num_1):
    #     res.append(values[i * num_2: (i + 1) * num_2].index(max(values[i * num_2: (i + 1) * num_2])))
    # return res

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
