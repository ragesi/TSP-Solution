# -*- coding: UTF-8 -*-
import random
import time

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
import qiskit.circuit.library as lib

import numpy as np
import math as m
import cmath as cm

from utils import display_result, NOT_gate, unitary_function as uf, util, execute
from dataset import test


class OptimalRoute:
    def __init__(self, node_num, node_list, dist_adj, total_qubit_num):
        """
        :param node_num: the number of cities in the route
        :param node_list: the list of cities' id,
                          the first element must be the start of route, and the last element must be the end
        :param dist_adj: the distance adjacency matrix of nodes, whose size is [node_num - 1, node_num - 1]
        """
        # TODO: 处理node_list的映射关系
        # the number of nodes in the route
        self.node_num = node_num
        self.node_list = node_list
        # self.dis_adj's size is [node_num - 1, 4], filling 0 to make-up, which excludes the last step's distance
        self.dist_adj = np.zeros((self.node_num - 1, self.node_num - 2))
        self.end_dists = np.zeros(self.node_num - 2)
        self.total_qubit_num = total_qubit_num

        # the number of choices for every step, excluding the start and the end
        self.choice_num = self.node_num - 2
        # the number of steps that have choices
        self.step_num = self.node_num - 2
        # the number of choice encoding bits in binary
        self.choice_bit_num = m.ceil(1.0 * m.log2(self.choice_num))
        self.qram_num = self.step_num * self.choice_bit_num
        # the precision of route distance
        self.precision = 6
        self.anc_num = self.choice_bit_num - 1
        self.buffer_num = max(self.precision, self.step_num)
        self.res_num = 3
        self.thresholds = []
        # self.illegal_choice_list = []
        # self.illegal_route_list = []

        self.qram = None
        self.buffer = None
        self.anc = None
        self.res = None
        self.cl = None
        self.qc = None

        self.init_dis_adj(dist_adj)
        self.init_circuit()
        # # get the parameter of algorithm, which is used in each iteration
        # self.get_illegal_choice_param()
        # self.get_illegal_route_param()

    def init_dis_adj(self, dist_adj):
        """
        the distance matrix's preprocessing
        :param dist_adj: matrix representing the distance between every city
        """
        # calculate the total distance
        total_distance = 0.0
        for i in np.arange(len(dist_adj)):
            if i > 0:
                dist_adj[i][i - 1] = max(dist_adj[i])
            if i < len(dist_adj) - 1:
                total_distance += max(dist_adj[i][i: -1])
        total_distance += max([row[-1] for row in dist_adj])

        # normalize the adjacency matrix to 2.0 * m.pi / (2 ** precision)
        base_num = 2 ** self.precision
        for i in np.arange(len(dist_adj)):
            for j in np.arange(len(dist_adj[i])):
                dist_adj[i][j] = round(1.0 * dist_adj[i][j] / total_distance * base_num) / base_num

        # make up the last step's distance
        for i in np.arange(len(dist_adj)):
            if i == 0:
                dist_adj[i].pop()
                continue
            self.end_dists[i - 1] = dist_adj[i].pop()
        # make up the rest of adjacency matrix
        self.dist_adj[:, :self.node_num - 2] = dist_adj

    def init_circuit(self):
        """
        init the quantum circuit for detecting the legal route, which has 4 parts
        1. choice register: store the choice of every step
        2. detail register: store the temporary info of processing
        3. ancilla register: to assist the process of algorithm
        4. result register: to record the result of every stage of algorithm
        """
        if self.qram_num + self.buffer_num + self.anc_num + self.res_num > self.total_qubit_num:
            raise ValueError("The number of nodes is too much!")
        # for the end node, it doesn't have choice; for the node_list[-2], it only has one choice for the end
        self.anc_num = max(self.anc_num, self.total_qubit_num - self.qram_num - self.buffer_num - self.res_num)
        print("qram_num: ", self.qram_num)
        print("buffer_num: ", self.buffer_num)
        print("anc_num: ", self.anc_num)
        print("res_num: ", self.res_num)

        self.qram = QuantumRegister(self.qram_num, name='qram')
        self.buffer = QuantumRegister(self.buffer_num, name='tmp_info')
        self.anc = QuantumRegister(self.anc_num, name='anc')
        self.res = QuantumRegister(4, name='res')
        self.cl = ClassicalRegister(self.qram_num + 1, name='c')
        self.qc = QuantumCircuit(self.qram, self.buffer, self.anc, self.res, self.cl)

        self.qc.h(self.qram)
        self.qc.x(self.res[-1])
        self.qc.h(self.res[-1])

    def check_choice_validity(self):
        """
        determine whether the choice of every step is less than the number of nodes that can be chosen
        :return: QuantumCircuit
        """
        qram = QuantumRegister(self.qram_num)
        buffer = QuantumRegister(self.step_num)
        anc = QuantumRegister(self.anc_num)
        res = QuantumRegister(1)
        qc = QuantumCircuit(qram, buffer, anc, res)

        # integer comparator
        for i in np.arange(self.step_num):
            qc.append(lib.IntegerComparator(self.choice_bit_num, self.choice_num, False),
                      [*qram[i * self.choice_bit_num: (i + 1) * self.choice_bit_num], buffer[i],
                       *anc[:(self.choice_bit_num - 1)]])
        qc.append(NOT_gate.custom_mcx(self.step_num, self.anc_num), [*buffer, *anc, *res])
        for i in np.arange(self.step_num - 1, -1, -1):
            qc.append(lib.IntegerComparator(self.choice_bit_num, self.choice_num, False).inverse(),
                      [*qram[i * self.choice_bit_num: (i + 1) * self.choice_bit_num], buffer[i],
                       *anc[:(self.choice_bit_num - 1)]])

        return qc

    def check_route_validity(self):
        qram = QuantumRegister(self.qram_num)
        buffer = QuantumRegister(self.step_num)
        anc = QuantumRegister(self.anc_num)
        res = QuantumRegister(1)
        qc = QuantumCircuit(qram, buffer, anc, res)

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

    def cal_distance_qpe(self):
        """
        to calculate the total distance of every route
        """
        qram = QuantumRegister(self.qram_num)
        buffer = QuantumRegister(self.precision)
        anc = QuantumRegister(self.anc_num)
        qc = QuantumCircuit(qram, buffer, anc)

        # QPE
        qc.h(buffer)
        # for every step
        for i in np.arange(self.step_num):
            if i == 0:
                qc.append(uf.QPE_U(self.precision, self.choice_bit_num, self.dist_adj[0], self.anc_num),
                          [*buffer, *qram[:self.choice_bit_num], *anc])
            else:
                qc.append(uf.custom_QPE_U(self.precision, self.choice_bit_num, self.anc_num, self.dist_adj[1:]),
                          [*buffer, *qram[(i - 1) * self.choice_bit_num: (i + 1) * self.choice_bit_num], *anc])
        # the last step which is not shown in the self.step_num
        qc.append(uf.QPE_U(self.precision, self.choice_bit_num, self.end_dists, self.anc_num),
                  [*buffer, *qram[-self.choice_bit_num:], *anc])
        qc.append(lib.QFT(self.precision, do_swaps=False, inverse=True), buffer)

        return qc

    def check_dist_below_threshold(self, threshold):
        """
        calculate the distance of every route, and determine whether it is less than the threshold
        :param threshold: integer
        """
        qram = QuantumRegister(self.qram_num)
        buffer = QuantumRegister(self.precision)
        anc = QuantumRegister(self.anc_num)
        res = QuantumRegister(1)
        qc = QuantumCircuit(qram, buffer, anc, res)

        # qc.append(self.cal_distance_qpe(), [*qram, *buffer, *anc, *res])
        # qc.append(lib.IntegerComparator(self.precision, threshold, geq=False),
        #           [*buffer, *res, *anc[:self.precision - 1]])
        qc.append(self.cal_distance_qpe(), [*qram, *buffer, *anc])
        qc.append(lib.IntegerComparator(self.precision, threshold, geq=False),
                  [*buffer, *anc[:self.precision]])
        qc.cx(anc[0], res[0])
        # qc.append(lib.IntegerComparator(self.precision, threshold, geq=False).inverse(),
        #           [*buffer, *anc[:self.precision]])
        # qc.append(self.cal_distance_qpe().inverse(), [*qram, *buffer, *anc])

        return qc

    def grover_operator(self, threshold):
        """
        an iteration of Grover algorithm
        """
        qram = QuantumRegister(self.qram_num)
        buffer = QuantumRegister(self.buffer_num)
        anc = QuantumRegister(self.anc_num)
        res = QuantumRegister(self.res_num)
        qc = QuantumCircuit(qram, buffer, anc, res)

        # three stage to determine whether the route is valid
        # qc.append(self.check_choice_validity(), [*qram, *buffer[:self.step_num], *anc, res[0]])
        qc.append(self.check_route_validity(), [*qram, *buffer[:self.step_num], *anc, res[0]])
        qc.append(self.check_dist_below_threshold(threshold), [*qram, *buffer[:self.precision], *anc, res[1]])

        # qc.cx(res[1], res[-1])
        qc.ccx(res[0], res[1], res[-1])

        qc.append(self.check_dist_below_threshold(threshold).inverse(), [*qram, *buffer[:self.precision], *anc, res[1]])
        qc.append(self.check_route_validity().inverse(), [*qram, *buffer[:self.step_num], *anc, res[0]])
        # qc.append(self.check_choice_validity().inverse(), [*qram, *buffer[:self.step_num], *anc, res[0]])

        # grover diffusion
        qc.append(uf.grover_diffusion(self.qram_num, self.anc_num), [*qram, *anc, res[-1]])

        return qc

    def grover(self, threshold, iter_num):
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

        for _ in np.arange(iter_num):
            qc.append(self.grover_operator(threshold), [*qram, *buffer, *anc, *res])

        qc.measure(qram, cl)
        # return execute.local_simulator(qc, 1000)
        return list(execute.local_simulator(qc, 1))[0]

    def cal_single_route_dist(self, route):
        dist = 0.0
        for i in np.arange(len(route)):
            if i == 0:
                dist += self.dist_adj[0][route[i]]
            else:
                dist += self.dist_adj[route[i - 1] + 1][route[i]]
        dist += self.end_dists[route[-1]]
        return dist

    def translate_route(self, bin_route):
        bin_route = bin_route[::-1]
        route = []
        for i in np.arange(self.step_num):
            route.append(int(bin_route[i * self.choice_bit_num: (i + 1) * self.choice_bit_num], 2))
        return route

    def QESA(self, threshold, min_iter_num):
        # TODO: 此处要调大alpha，在6个节点的规模下，每次增长1就可以，要不然增长幅度太小；如果接下来规模变大，可以考虑使用比例调整
        alpha = 6.0 / 5
        max_iter_num = min_iter_num

        iter_num_bound = m.ceil(4.5 * m.sqrt(2 ** self.qram_num))
        is_finish = True
        new_threshold = 0.
        new_route = None
        for _ in np.arange(iter_num_bound):
            iter_num = random.randint(int(min_iter_num), int(max_iter_num))
            new_route = self.grover(threshold, iter_num)
            new_route = self.translate_route(new_route)
            new_threshold = self.cal_single_route_dist(new_route)

            if new_threshold < threshold:
                is_finish = False
                break
            else:
                max_iter_num = min(alpha * max_iter_num, m.sqrt(2 ** self.qram_num))

        return new_route, new_threshold, is_finish, max_iter_num

    def QMSA(self):
        min_iter_num = 1.
        for i in np.arange(self.step_num, 0, -1):
            min_iter_num *= m.sqrt(1.0 * i / (2 ** self.choice_bit_num))
        min_iter_num = m.pi / 4.0 / m.asin(min_iter_num)
        min_iter_num = max(1.0, min_iter_num)
        print("min_iter_num: ", min_iter_num)

        threshold = 0.
        for i in np.arange(len(self.dist_adj) - 1):
            threshold += self.dist_adj[i][i]
        threshold += self.end_dists[-1]
        threshold = min(threshold, 1.0 - (1.0 / 64))
        threshold *= 2 ** self.precision
        print("threshold: ", threshold)

        route = None

        while True:
            tmp_route, tmp_threshold, is_finish, tmp_iter_num = self.QESA(threshold, min_iter_num)
            if not is_finish:
                threshold = tmp_threshold
                route = tmp_route
                min_iter_num = tmp_iter_num
            else:
                break

        return route


if __name__ == '__main__':
    test_adj = test.test_for_5
    test = OptimalRoute(5, [0, 1, 2, 3, 4], test_adj, 27)
    print(test.dist_adj)
    print(test.end_dists)
    # test.qc.append(test.check_route_validity(), [*test.qram, *test.buffer[:test.step_num], *test.anc, test.res[0]])
    # test.qc.append(test.check_dist_below_threshold(61), [*test.qram, *test.buffer[:test.precision], *test.anc, test.res[1]])
    # test.qc.measure([*test.qram, test.res[1]], test.cl)
    # output = execute.local_simulator(test.qc, 1000)

    # qram = QuantumRegister(2)
    # buffer = QuantumRegister(6)
    # anc = QuantumRegister(16)
    # res = QuantumRegister(3)
    # cl = ClassicalRegister(6)
    # qc = QuantumCircuit(qram, buffer, anc, res, cl)
    # qc.x(qram[1])
    # qc.append(test.cal_distance_qpe(), [*qram, *buffer, *anc])
    # qc.measure(buffer, cl)
    # output = execute.local_simulator(qc, 1000)
    # print(output)
    # num = 0
    # for item in output.items():
    #     if item[0][0] == '1':
    #         print(item)
    #         num += 1
    # print(num)

    # start_time = time.time()
    # output = test.grover(61, 1)
    # end_time = time.time()
    # print("time: ", end_time - start_time)
    # print(output)
    start_time = time.time()
    route = test.QMSA()
    end_time = time.time()
    print("time: ", end_time - start_time)
    print(route)
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
