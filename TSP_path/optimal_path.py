# -*- coding: UTF-8 -*-
import random
import time

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler, Options
import qiskit.circuit.library as lib

import numpy as np
import math as m
import cmath as cm

from utils import NOT_gate, util
from dataset import test


class OptimalPath:
    def __init__(self, point_num, points, total_qubit_num):
        """
        :param point_num: the number of cities in the route
        :param points: coordinates of all nodes
        """
        self.points = points
        self.point_num = point_num
        self.total_qubit_num = total_qubit_num
        # the number of choices for every step, excluding the start and the end
        self.choice_num = point_num - 2
        # the number of steps that have choices
        self.step_num = point_num - 2
        # the number of choice encoding bits in binary
        self.choice_bit_num = m.ceil(1.0 * m.log2(self.choice_num))
        self.qram_num = self.step_num * self.choice_bit_num
        # the precision of route distance
        self.precision = 6
        self.anc_num = self.precision - 1
        self.buffer_num = max(self.precision, self.step_num)
        self.res_num = 3

        # the parameters for the process of Grover algorithm
        self.threshold = 0.0
        self.path = [i for i in np.arange(self.step_num)]
        self.grover_iter_min_num = 1.0
        self.grover_iter_max_num = 1.0
        self.alpha = 6.0 / 5
        self.grover_repeat_num = 0

        # the distance adjacency
        self.dist_adj = np.zeros((point_num - 1, 2 ** self.choice_bit_num))
        self.end_dists = np.zeros(2 ** self.choice_bit_num)

        # run on the IBM Quantum Platform
        self.session = None
        self.sampler = None
        self.job = None

        # initialize all parameters for Grover algorithm
        self.init_grover_param()

    def init_dist_adj(self):
        """
        the distance matrix's preprocessing
        """
        dist_adj = np.zeros((self.point_num - 1, self.point_num - 1))
        order_dists = []
        max_dist = 0
        # TODO: 此处需要重新设计距离的计算方法，保证真实距离和最终的结果是线性负相关的
        for i in range(self.point_num - 1):
            for j in range(i, self.point_num - 1):
                if i == 0 and j == self.point_num - 2:
                    continue
                dist_adj[i][j] = abs(self.points[j + 1][0] - self.points[i][0]) + abs(
                    self.points[j + 1][1] - self.points[i][1])
                max_dist = max(max_dist, dist_adj)
                # if i > 0 and j < self.point_num - 2:
                #     dist_adj[j + 1][i - 1] = dist_adj[i][j]
                    # order_dists.append(dist_adj[i][j])

        for i in range(self.point_num - 1):
            for j in range(i, self.point_num - 1):
                if i == 0 and j == self.point_num - 2:
                    continue
                dist_adj[i][j] = max_dist - dist_adj

        print("dist_adj: ", dist_adj)

        # calculate the max distance
        order_dists.sort(reverse=True)
        max_dist = max(dist_adj[0][:-1])
        for i in range(self.step_num - 1):
            max_dist += order_dists[i]
        max_dist += max([row[-1] for row in dist_adj]) + 1

        # normalize the adjacency matrix to 2.0 * m.pi / (2 ** precision)
        base_num = 2 ** self.precision
        for i in np.arange(len(dist_adj)):
            for j in np.arange(len(dist_adj[i])):
                dist_adj[i][j] = round(1.0 * dist_adj[i][j] / max_dist * base_num) / base_num

        # make up the last step's distance
        for i in np.arange(len(dist_adj)):
            if i == 0:
                continue
            self.end_dists[i - 1] = dist_adj[i][-1]
        # make up the rest of adjacency matrix
        self.dist_adj[:, :self.choice_num] = dist_adj[:, :self.choice_num]

    def init_grover_param(self):
        if self.point_num < 4:
            return

        # adjust the number of bits of every quantum register
        if self.qram_num + self.buffer_num + self.anc_num + self.res_num > self.total_qubit_num:
            raise ValueError("The number of nodes is too much!")
        self.anc_num = max(self.anc_num, self.total_qubit_num - self.qram_num - self.buffer_num - self.res_num)
        print("qram_num: ", self.qram_num)
        print("buffer_num: ", self.buffer_num)
        print("anc_num: ", self.anc_num)
        print("res_num: ", self.res_num)

        self.init_dist_adj()

        # initialize the threshold
        for i in np.arange(len(self.dist_adj) - 1):
            self.threshold += self.dist_adj[i][i]
        self.threshold += self.end_dists[self.choice_num - 1]
        self.threshold = min(self.threshold, 1.0 - (1.0 / 64))
        self.threshold *= 2 ** self.precision
        print("threshold: ", self.threshold)

        # initialize the number of iterations for Grover algorithm
        for i in np.arange(self.step_num, 0, -1):
            self.grover_iter_min_num *= m.sqrt(1.0 * i / (2 ** self.choice_bit_num))
        self.grover_iter_min_num = m.pi / 4.0 / m.asin(self.grover_iter_min_num)
        self.grover_iter_min_num = max(1.0, self.grover_iter_min_num)
        self.grover_iter_max_num = self.grover_iter_min_num
        print("iter_num: ", self.grover_iter_min_num)

        # initialize the number of repetitions of Grover algorithm
        self.grover_repeat_num = round(m.log(m.sqrt(m.factorial(self.choice_num)), self.alpha))

        # get connection to IBM Quantum Platform
        # options = Options()
        # options.optimization_level = 0
        # options.resilience_level = 1
        # service = QiskitRuntimeService()
        # # backend = 'ibmq_qasm_simulator'
        # # backend = 'simulator_statevector'
        # backend = 'ibm_brisbane'
        # self.session = Session(service=service, backend=backend)
        # self.sampler = Sampler(session=self.session, options=options)

    def check_route_validity(self):
        qram = QuantumRegister(self.qram_num)
        buffer = QuantumRegister(self.step_num)
        anc = QuantumRegister(self.anc_num)
        res = QuantumRegister(1)
        qc = QuantumCircuit(qram, buffer, anc, res, name='check_route_validity')

        for i in np.arange(self.step_num):
            for j in np.arange(self.choice_num):
                qc.append(NOT_gate.equal_to_int_NOT(j, self.choice_bit_num, self.anc_num),
                          [*qram[i * self.choice_bit_num: (i + 1) * self.choice_bit_num], *anc, buffer[j]])
        qc.append(NOT_gate.custom_mcx(self.step_num, self.anc_num), [*buffer, *anc, *res])
        for i in np.arange(self.step_num - 1, -1, -1):
            for j in np.arange(self.choice_num - 1, -1, -1):
                qc.append(NOT_gate.equal_to_int_NOT(j, self.choice_bit_num, self.anc_num).inverse(),
                          [*qram[i * self.choice_bit_num: (i + 1) * self.choice_bit_num], *anc, buffer[j]])

        return qc

    # the recursion method of QPE
    # def qpe_u_operator(self, dists: np.ndarray, target_num: int, anc_num: int) -> QuantumCircuit:
    #     control = QuantumRegister(1)
    #     target = QuantumRegister(target_num)
    #     anc = QuantumRegister(anc_num)
    #     qc = QuantumCircuit(control, target, anc)
    #
    #     if target_num < 2:
    #         qc.cp(2.0 * m.pi * (dists[1] - dists[0]), control, target[0])
    #         qc.p(2.0 * m.pi * dists[0], control)
    #         return qc
    #     elif target_num == 2:
    #         qc.cp(2.0 * m.pi * (dists[2] - dists[0]), control, target[1])
    #         qc.p(2.0 * m.pi * dists[0], control)
    #         qc.cp(2.0 * m.pi * (dists[1] - dists[0]), control, target[0])
    #
    #         delta_dists = 2.0 * m.pi * (dists[3] - dists[1] - dists[2] + dists[0])
    #         qc.ccx(control, target[1], anc[0])
    #         qc.cp(delta_dists, anc[0], target[0])
    #         qc.ccx(control, target[1], anc[0])
    #         return qc
    #     else:
    #         for i in np.arange(2 ** (target_num - 2)):
    #             reference_state = i + 2 ** target_num
    #             qc.append(NOT_gate.equal_to_int_NOT(reference_state, target_num - 1, anc_num - 1),
    #                       [*target[2:], control[0], *anc[1:], anc[0]])
    #
    #             qc.append(self.qpe_u_operator(dists[i * 4: (i + 1) * 4], 2, anc_num - 1), [anc[0], target[:2], anc[1:]])
    #
    #             qc.append(NOT_gate.equal_to_int_NOT(reference_state, target_num - 1, anc_num - 1).inverse(),
    #                       [*target[2:], control[0], *anc[1:], anc[0]])
    #
    # def qpe_u(self, dists: np.ndarray) -> QuantumCircuit:
    #     control = QuantumRegister(self.precision)
    #     target = QuantumRegister(self.choice_bit_num)
    #     anc = QuantumRegister(self.anc_num)
    #     qc = QuantumCircuit(control, target, anc)
    #
    #     for i in np.arange(self.precision):
    #         for _ in np.arange(2 ** (self.precision - i - 1)):
    #             qc.append(self.qpe_u_operator(dists, self.choice_bit_num, self.anc_num), [control[i], *target, *anc])
    #
    #     return qc
    #
    # def custom_qpe_u(self, dist_adj: np.ndarray) -> QuantumCircuit:
    #     control = QuantumRegister(self.precision)
    #     source = QuantumRegister(self.choice_bit_num)
    #     target = QuantumRegister(self.choice_bit_num)
    #     anc = QuantumRegister(self.anc_num)
    #     qc = QuantumCircuit(control, source, target, anc)
    #
    #     for i in np.arange(len(dist_adj)):
    #         for j in np.arange(self.precision):
    #             ref_state = i + 2 ** self.choice_bit_num
    #             qc.append(NOT_gate.equal_to_int_NOT(ref_state, self.choice_bit_num + 1, self.anc_num - 1),
    #                       [*source, control[j], *anc[1:], anc[0]])
    #
    #             for _ in np.arange(2 ** (self.precision - j - 1)):
    #                 qc.append(self.qpe_u_operator(dist_adj[i], self.choice_bit_num, self.anc_num - 1),
    #                           [anc[0], *target, *anc[1:]])
    #
    #             qc.append(NOT_gate.equal_to_int_NOT(ref_state, self.choice_bit_num + 1, self.anc_num - 1).inverse(),
    #                       [*source, control[j], *anc[1:], anc[0]])
    #
    #     return qc

    def qpe_u(self, dists: np.ndarray) -> QuantumCircuit:
        control = QuantumRegister(self.precision)
        target = QuantumRegister(self.choice_bit_num)
        anc = QuantumRegister(self.anc_num)
        qc = QuantumCircuit(control, target, anc)

        for i in np.arange(len(dists)):
            qc.append(NOT_gate.equal_to_int_NOT(i, self.choice_bit_num, self.anc_num - 1), [*target, *anc[1:], anc[0]])

            for j in np.arange(self.precision):
                for _ in np.arange(2 ** (self.precision - j - 1)):
                    qc.cp(2.0 * m.pi * dists[i], control[j], anc[0])

            qc.append(NOT_gate.equal_to_int_NOT(i, self.choice_bit_num, self.anc_num - 1).inverse(),
                      [*target, *anc[1:], anc[0]])

        return qc

    def custom_qpe_u(self, dist_adj: np.ndarray) -> QuantumCircuit:
        control = QuantumRegister(self.precision)
        source = QuantumRegister(self.choice_bit_num)
        target = QuantumRegister(self.choice_bit_num)
        anc = QuantumRegister(self.anc_num)
        qc = QuantumCircuit(control, source, target, anc)

        for i in np.arange(len(dist_adj)):
            qc.append(NOT_gate.equal_to_int_NOT(i, self.choice_bit_num, self.anc_num - 2), [*source, *anc[2:], anc[0]])

            for j in np.arange(len(dist_adj[i])):
                qc.append(NOT_gate.equal_to_int_NOT(j, self.choice_bit_num, self.anc_num - 2),
                          [*target, *anc[2:], anc[1]])

                for k in np.arange(self.precision):
                    qc.ccx(anc[0], control[k], anc[2])
                    for _ in np.arange(2 ** (self.precision - k - 1)):
                        qc.cp(2.0 * m.pi * dist_adj[i][j], anc[2], anc[1])
                    qc.ccx(anc[0], control[k], anc[2])

                qc.append(NOT_gate.equal_to_int_NOT(j, self.choice_bit_num, self.anc_num - 2).inverse(),
                          [*target, *anc[2:], anc[1]])

            qc.append(NOT_gate.equal_to_int_NOT(i, self.choice_bit_num, self.anc_num - 2).inverse(),
                      [*source, *anc[2:], anc[0]])

        return qc

    def grover_diffusion(self):
        qram = QuantumRegister(self.qram_num)
        anc = QuantumRegister(self.anc_num)
        res = QuantumRegister(1)
        qc = QuantumCircuit(qram, anc, res, name='grover_diffusion')

        qc.h(qram)
        qc.x(qram)
        qc.append(NOT_gate.custom_mcx(self.qram_num, self.anc_num), [*qram, *anc, *res])
        qc.x(qram)
        qc.h(qram)

        return qc

    def cal_path_dist(self):
        """
        to calculate the total distance of every route
        """
        qram = QuantumRegister(self.qram_num)
        buffer = QuantumRegister(self.precision)
        anc = QuantumRegister(self.anc_num)
        qc = QuantumCircuit(qram, buffer, anc, name='cal_distance_qpe')
        # QPE
        qc.h(buffer)
        # for every step
        for i in np.arange(self.step_num):
            if i == 0:
                qc.append(self.qpe_u(self.dist_adj[0]), [*buffer, *qram[:self.choice_bit_num], *anc])
            else:
                qc.append(self.custom_qpe_u(self.dist_adj[1:]),
                          [*buffer, *qram[(i - 1) * self.choice_bit_num: (i + 1) * self.choice_bit_num], *anc])
        # the last step which is not shown in the self.step_num
        qc.append(self.qpe_u(self.end_dists), [*buffer, *qram[-self.choice_bit_num:], *anc])
        qc.append(lib.QFT(self.precision, do_swaps=False, inverse=True), buffer)
        return qc

    def build_partial_circuit(self):
        qram = QuantumRegister(self.qram_num)
        buffer = QuantumRegister(self.buffer_num)
        anc = QuantumRegister(self.anc_num)
        res = QuantumRegister(self.res_num)
        qc_start = QuantumCircuit(qram, buffer, anc, res, name='qc_start')
        qc_end = QuantumCircuit(qram, buffer, anc, res, name='qc_end')

        qc_start.append(self.check_route_validity(), [*qram, *buffer[:self.step_num], *anc, res[0]])
        qc_start.append(self.cal_path_dist(), [*qram, *buffer[:self.precision], *anc])

        qc_end.append(self.cal_path_dist().inverse(), [*qram, *buffer[:self.precision], *anc])
        qc_end.append(self.check_route_validity().inverse(), [*qram, *buffer[:self.step_num], *anc, res[0]])
        qc_end.append(self.grover_diffusion(), [*qram, *anc, res[-1]])

        return qc_start, qc_end

    def cal_single_route_dist(self, route):
        dist = 0.0
        for i in np.arange(len(route)):
            if route[i] >= self.choice_num or (i > 0 and route[i - 1] >= self.choice_num):
                continue
            elif i == 0:
                dist += self.dist_adj[0][route[i]]
            else:
                dist += self.dist_adj[route[i - 1] + 1][route[i]]
        dist += self.end_dists[route[-1]] if route[-1] < self.choice_num else 0
        dist *= 2 ** self.precision
        return dist

    def translate_route(self, bin_route):
        route = []
        for i in np.arange(self.step_num):
            route.insert(0, int(bin_route[i * self.choice_bit_num: (i + 1) * self.choice_bit_num], 2))
        return route

    def async_grover(self):
        qram = QuantumRegister(self.qram_num)
        buffer = QuantumRegister(self.buffer_num)
        anc = QuantumRegister(self.anc_num)
        res = QuantumRegister(self.res_num)
        cl = ClassicalRegister(self.qram_num)
        qc = QuantumCircuit(qram, buffer, anc, res, cl)
        # initialization
        qc.h(qram)
        qc.x(res[-1])
        qc.h(res[-1])

        max_iter_bound = m.pi / 4.0 * m.sqrt(2 ** self.qram_num)

        qc_start, qc_end = self.build_partial_circuit()
        # print(qc_start)
        # print(qc_end)

        if self.job is not None:
            # output = self.job.result().quasi_dists[0]
            output = self.job.result().get_counts()
            output = sorted(output.items(), key=lambda item: item[1], reverse=True)
            # output = util.int_to_binary(output[0][0], self.qram_num)
            # new_path = self.translate_route(output)
            new_path = self.translate_route(output[0][0])
            new_threshold = self.cal_single_route_dist(new_path)
            print("new_path: ", new_path)

            if new_threshold > self.threshold:
                self.threshold = new_threshold
                self.path = new_path
                # self.grover_iter_min_num = min(self.grover_iter_min_num * 1.4, max_iter_bound)
                self.grover_iter_min_num = 1.0 / 2 * (self.grover_iter_max_num + self.grover_iter_min_num)
                # self.grover_iter_max_num = max(self.grover_iter_min_num, self.grover_iter_max_num)
                self.grover_repeat_num = round(m.log(m.sqrt(m.factorial(self.choice_num)), self.alpha))
                print("new_threshold: ", new_threshold)
            else:
                self.grover_repeat_num -= 1
                self.grover_iter_max_num = min(self.alpha * self.grover_iter_max_num, max_iter_bound)
        if self.grover_repeat_num == 0:
            # self.session.close()
            return

        cur_iter_num = random.randint(int(self.grover_iter_min_num), int(self.grover_iter_max_num))
        print("cur_iter_num: ", cur_iter_num)
        for _ in range(cur_iter_num):
            qc.append(qc_start, [i for i in range(self.total_qubit_num)])
            qc.append(lib.IntegerComparator(self.precision, self.threshold, geq=True),
                      [*buffer, res[1], *anc[:self.precision - 1]])
            qc.ccx(res[0], res[1], res[-1])
            qc.append(lib.IntegerComparator(self.precision, self.threshold, geq=True).inverse(),
                      [*buffer, res[1], *anc[:self.precision - 1]])
            qc.append(qc_end, [i for i in range(self.total_qubit_num)])

        # print(qc)
        print("depth: ", qc.depth())
        qc.measure(qram, cl)
        backend = Aer.backends(name='qasm_simulator')[0]
        self.job = execute(qc, backend, shots=1000)
        # self.job = self.sampler.run(circuits=qc, shots=1000)
        self.async_grover()

    def main(self):
        if self.point_num < 4:
            return [i for i in range(self.point_num)]

        self.async_grover()
        path = [0]
        for i in range(len(self.path)):
            path.append(self.path[i] + 1)
        path.append(len(path))
        return path

        # path = [self.points[0]]
        # for i in range(len(self.path)):
        #     path.append(self.points[self.path[i] + 1])
        # path.append(self.points[-1])
        # return path


if __name__ == '__main__':
    # test_adj = test.test_for_5
    test_points = test.point_test_for_4
    test = OptimalPath(4, test_points, 27)
    print(test.dist_adj)
    print(test.end_dists)
    # test.qc.append(test.check_route_validity(), [*test.qram, *test.buffer[:test.step_num], *test.anc, test.res[0]])
    # test.qc.append(test.check_dist_below_threshold(53), [*test.qram, *test.buffer[:test.precision], *test.anc, test.res[1]])
    # test.qc.measure([*test.qram, test.res[0]], test.cl)
    # output = execute.local_simulator(test.qc, 1000)

    # dists = [0.390625, 0., 0., 0.]
    # qram = QuantumRegister(6)
    # buffer = QuantumRegister(6)
    # anc = QuantumRegister(12)
    # res = QuantumRegister(3)
    # cl1 = ClassicalRegister(6)
    # # cl2 = ClassicalRegister(6)
    # qc = QuantumCircuit(qram, buffer, anc, res, cl1)
    # qc.x([qram[2], qram[5]])
    # qc.append(test.cal_distance_qpe(), [*qram, *buffer, *anc])
    # # qc.measure(qram, cl1)
    # qc.measure(buffer, cl1)
    # output = execute.local_simulator(qc, 10)
    # print(output)
    # num = 0
    # for item in output.items():
    #     if item[0][0] == '1':
    #         print(item)
    #         num += 1
    # print(num)

    # start_time = time.time()
    # output = test.grover(48, 2)
    # end_time = time.time()
    # print("time: ", end_time - start_time)
    # output = sorted(output.items(), key=lambda item: item[1], reverse=True)
    # print(output)

    # start_time = time.time()
    # test.async_grover()
    # end_time = time.time()
    # print("time: ", end_time - start_time)
    # path = test.path
    # print(path)

    # test.qc.measure([*test.qram, test.res[0]], test.cl)
    # output = execute.local_simulator(test.qc, 1000)
    # output = display_result.Measurement(test.qc, return_M=True, print_M=False, shots=2000)
    # output = execute.sampler_run('simulator_mps', 1, 2, test.qc, 1000)
    # output = dict(output)
    # output = {util.int_to_binary(key, test.qram_num + 1): val for key, val in output.items()}
    # output = sorted(output.items(), key=lambda item: item[1], reverse=True)
    # print(output)
    # num = 0
    # for item in output.items():
    #     if item[0][0] == '1':
    #         print(item)
    #         num += 1
    # print(num)
